# LLM error correction (GER) for filler-HuBERT SA8+PE decodes

**Question:** can a text-only LLM (plain inference, no SLAM-ASR) take the SA8+PE
model's output and correct its mistakes?

**Answer: yes — but only on the *cleaned* text, and only with a gate.**
Corpus WER on full test-clean drops from **12.04% → 9.64%** (19.9% relative)
using zero-shot Qwen2.5-7B-Instruct + OOV gate: 438 utts improved, 73
worsened, 2109 unchanged; the gate rejected 521 needless LLM edits. Feeding the raw token stream
(`<s> m i s t e r | ... </s> <fill> ...`) directly to the LLM makes things
*worse* (11.2% → 24.3% on a 100-utt sample): on long inputs the model echoes
the spaced-char format back instead of joining letters, and it hallucinates
over garbled stretches. Format stripping must stay deterministic; the LLM only
does linguistic repair.

## Setup

- Decodes: `../strip_variants/test_clean.decodes.txt` — SA-8+PE run
  (`full960_sa8_pe_fps25_lin`, best.pt), test-clean, 2620 utts, fillrun3 cut,
  baseline WER 12.04%.
- LLM: `Qwen/Qwen2.5-7B-Instruct` (bf16, single RTX 6000 Ada), greedy,
  zero-shot prompt (see `correct_llm.py`), batch 32. Full test set ≈ 6 min.
- Post-normalization: lowercase, strip non-`[a-z0-9' ]`.
- Metric: `jiwer` corpus WER.

## Key findings (300-utt pilot)

| variant | WER | utts improved | utts worsened |
|---|---|---|---|
| no correction | 11.01% | – | – |
| LLM, ungated (v1 prompt) | 8.95% | 66 | 47 |
| LLM, ungated (conservative v2 prompt) | 9.40% | 67 | 58 |
| **LLM + OOV gate** | **8.47%** | **62** | **8** |
| LLM on RAW token stream (100 utts) | 24.25% | 15 | 45 |

1. **On erroneous utts the LLM is genuinely good**: 66 improved vs 11 worsened
   (ungated pilot). It reliably repairs non-word misspellings and moderately
   garbled stretches ("came sso far intotthe thic forett" → "came so far into
   the thick forest", 0.48 → 0.00 WER).
2. **The damage comes from already-perfect utts**: ungated, the LLM edited 36
   of 177 perfect transcripts — paraphrasing valid words ("arcadian" →
   "idyllic"), modernizing archaic book English ("ye" → "you"), re-spacing
   names ("hawkeye" → "hawk eye"). Prompt tuning did NOT fix this (v2 was
   worse); a hard gate did.
3. **OOV gate**: accept the LLM's edit only if the hypothesis contains a token
   outside a 142k-word vocab (LibriSpeech train-960 transcript vocab ∪
   /usr/share/dict/words → `gate_vocab.txt`). On the full set the gate fires on
   740/2620 utts, catches 695/916 truly erroneous ones (76%), and exposes only
   45 clean utts to the LLM. The 221 missed errors are real-word substitutions
   (e.g. "baddleax"/"battleax"-type, "colours"/"colors") that a text-only LLM
   mostly cannot fix without audio.
4. **Fully collapsed utterances (WER ≈ 1.0, repeated-char garbage) are
   unrecoverable** — no linguistic signal survives; the LLM hallucinates
   fluent but wrong text. These bound the achievable gain.

## Full test-clean result (2620 utts, gated)

| | WER | improved | worsened | unchanged |
|---|---|---|---|---|
| before | 12.04% | – | – | – |
| after LLM+gate | **9.64%** | 438 | 73 | 2109 |

Runtime: 323 s on one RTX 6000 Ada (batch 32, bf16).
Raw numbers in `corrected.clean.gated.full.summary.json`.

## Files

- `correct_llm.py` — main script (`--mode clean|raw`, `--gate`, `--n`)
- `gate_vocab.txt` — 142,223-word gating vocabulary
- `corrected.clean.gated.full.jsonl` — per-utt ref/hyp/corrected + WERs
- `corrected.raw.n100.jsonl` — raw-format experiment
- pilots: `corrected.clean.n300.jsonl` (v1), `corrected.clean.v2.n300.jsonl`,
  `corrected.clean.gated.n300.jsonl`

## Next ideas

- N-best / uncertainty from the ASR (per-frame logits) in the prompt — the
  standard GER setup; helps with real-word errors.
- Few-shot exemplars of typical SA8+PE error patterns in the prompt.
- A larger LLM (Qwen2.5-72B / API model) for the gated subset only (740 utts).
- Confidence-gate from ASR posteriors instead of the OOV vocab gate.
