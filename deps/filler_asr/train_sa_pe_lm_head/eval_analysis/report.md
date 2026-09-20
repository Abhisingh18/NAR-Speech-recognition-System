# Filler-ASR eval analysis — duration-ceiling (n_keep) capping

**Inputs:** the 3 W&B sample dumps (`ep8.csv`, `4ep.csv`, `ep6.csv`) = the `dump_k=10`
shortest dev-clean clips (<10 s) decoded by three runs.

**Run identity** (recovered from the `[setup]` lines of the 30-ep logs; CSV filenames are misleading):

| csv | run | sa_layers | trainable | step | logged dev WER (500 utts, uncapped) |
|-----|-----|-----------|-----------|------|------|
| ep8.csv | SA-8 | 8 | 157M | 24000 | 0.300 |
| 4ep.csv | SA-6 | 6 | 118M | 30000 | 0.340 |
| ep6.csv | SA-4 | 4 |  79M | 24000 | 0.382 |

## The bug being fixed
`evaluate()` (train.py:134-139) and `infer.py:114` decode `pred` over the **whole padded
batch**, then strip specials+`<fill>`. Frames **past the duration ceiling**
`n_keep = ceil(dur·25)` (the right-pad region) are not all `<fill>`: when the model emits
real chars there they survive stripping and append as trailing garbage → **inflated WER/CER**.
`full_frames` (the graded keep-window) does NOT contain this garbage, so it is the correct,
duration-capped output. Fix = cap the decode at the first `</s>` / at `n_keep`.

## Result (10-utt subset; capping at </s>)

| run | WER_old | WER_cap | Δrel | CER_old | CER_cap | SER_old | SER_cap |
|-----|---------|---------|------|---------|---------|---------|---------|
| SA-4 | 0.320 | 0.235 | -26% | 0.211 | 0.063 | 1.00 | 1.00 |
| SA-6 | 0.281 | 0.275 | -2%  | 0.088 | 0.067 | 0.90 | 0.90 |
| SA-8 | 0.359 | **0.098** | **-73%** | 0.475 | **0.025** | 0.70 | 0.30 |

## Insights
1. **The leak inverts the ranking.** Uncapped, the *best* model (SA-8) looks the *worst*
   (0.359). Capped, SA-8 is far ahead (0.098) — matching its lower logged dev WER. Any model
   selection / early-stopping on the uncapped WER is biased.
2. **Leak magnitude is model-dependent, not data-dependent.** Same 10 clips → identical
   padding. SA-8 hallucinates chars in the pad region (u3: +19 garbage words, CER 0.475);
   SA-6 @30k has learned to emit `<fill>` past content, so it barely leaks (Δ≈0).
3. **WER vs CER sensitivity.** Garbage only becomes extra *words* when a `|` is predicted in
   the pad region (SA-8 u3/u4). When it's glued to the last token (SA-6 u3 `tthaaa`+`eerrr…`)
   WER is unchanged but CER still suffers — so **CER is the more honest leak detector**.
4. **Residual capped errors are genuine acoustic/spelling errors** on the long, hard utts
   (u3 leighton/ithaca, u4 luminous/criticisms), not artefacts. Short utts (u5,u7,u9) are
   near-perfect for SA-8.
5. Training curves: SA-8 best content-frame acc (0.83) and lowest WER but **bumpy** (0.254@22k
   → 0.300@24k). SA-6 has loss spikes (1.45@22k); SA-4 content-acc collapses ~16-18k. Depth
   helps (8>6>4) but bigger heads are less stable.

## Recommended code fix (true n_keep-capped WER)
In `evaluate()` / `infer.py`, replace the batch-wide decode with a per-utt EOS-capped decode:
```python
eos = tok.eos_token_id
def cap_decode(p1d):                       # p1d = pred[i] (1-D frame ids)
    hit = (p1d == eos).nonzero()
    if len(hit): p1d = p1d[:hit[0, 0]]     # stop at first </s>
    return tok.decode(p1d, skip_special_tokens=True, group_tokens=False).strip()
hyp = cap_decode(pred[i])
```
(or restrict to `pred[i][keep]` first, then cut at `</s>`).
