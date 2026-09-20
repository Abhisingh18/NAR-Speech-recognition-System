"""
Mask-predict decoding, decoupled from fairseq and from the [CLS] length head.

Faithful port of facebookresearch/Mask-Predict
  fairseq/strategies/mask_predict.py  ::  MaskPredict.generate
but:
  * NO length token / [CLS] head  -> you pass target `lengths` in yourself.
  * NO fairseq  -> you pass a plain `decoder_forward(tokens, memory) -> logits`.

Requirements on your model:
  * decoder MUST be bidirectional (remove the causal self-attention mask).
  * `<mask>` must be a real id in the embedding table + output vocab.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def mask_predict_decode(decoder_forward, memory, lengths, mask_id, pad_id,
                        iterations=10, verbose=False, id2tok=None):
    """
    decoder_forward : callable(tgt_tokens[B,L], memory) -> logits[B,L,V]
    memory          : whatever your encoder produced (opaque; passed straight through)
    lengths         : LongTensor[B]  target length N per example  (YOU decide these)
    mask_id, pad_id : int
    iterations      : T (constant number of mask-predict cycles)
    returns:
        tokens [B, Lmax]  (pad beyond each row's length)
        score  [B]        average log-prob (use to pick among length candidates)
    """
    device = lengths.device
    B = lengths.size(0)
    Lmax = int(lengths.max().item())

    # ---- build the all-<mask> canvas; pad positions beyond each row's length ----
    tgt = torch.full((B, Lmax), mask_id, dtype=torch.long, device=device)
    positions = torch.arange(Lmax, device=device).unsqueeze(0)       # [1, Lmax]
    pad_mask = positions >= lengths.unsqueeze(1)                      # True where pad
    tgt[pad_mask] = pad_id

    # ---- t = 0 : predict everything from all masks (pure non-autoregressive) ----
    tok, prob = _step(decoder_forward, tgt, memory)
    tgt = torch.where(pad_mask, tgt, tok)
    prob = prob.masked_fill(pad_mask, 1.0)          # pad prob=1 -> never chosen as "worst"
    if verbose: _show(0, None, tgt, pad_mask, id2tok)

    # ---- t = 1 .. T-1 : mask the least-confident, re-predict ----
    for t in range(1, iterations):
        num_mask = (lengths.float() * (1.0 - t / iterations)).long().clamp(min=1)
        worst = _select_worst(prob, num_mask, pad_mask)              # [B,Lmax] bool
        tgt = tgt.masked_fill(worst, mask_id)

        tok, new_prob = _step(decoder_forward, tgt, memory)
        # update ONLY masked slots; kept slots retain old token + old (stale) prob
        tgt = torch.where(worst, tok, tgt)
        prob = torch.where(worst, new_prob, prob)
        prob = prob.masked_fill(pad_mask, 1.0)
        if verbose: _show(t, worst, tgt, pad_mask, id2tok)

    valid = ~pad_mask
    score = (prob.clamp_min(1e-9).log() * valid).sum(-1) / lengths.float()
    return tgt, score


def _step(decoder_forward, tgt, memory):
    logits = decoder_forward(tgt, memory)           # [B, L, V]
    probs = F.softmax(logits, dim=-1)
    prob, tok = probs.max(dim=-1)                   # greedy token + its confidence
    return tok, prob


def _select_worst(prob, num_mask, pad_mask):
    """Boolean mask of the num_mask[b] lowest-prob (non-pad) positions per row."""
    B, L = prob.size()
    p = prob.masked_fill(pad_mask, float("inf"))    # pad can't be 'worst'
    worst = torch.zeros(B, L, dtype=torch.bool, device=prob.device)
    for b in range(B):
        k = int(num_mask[b].item())
        idx = p[b].topk(k, largest=False, sorted=False).indices
        worst[b, idx] = True
    return worst


# ---------- optional: pick best among several length candidates ----------
@torch.no_grad()
def mask_predict_with_length_beam(decoder_forward, memory_expand, cand_lengths,
                                  mask_id, pad_id, iterations=10):
    """
    cand_lengths : LongTensor[B, ell]   ell length candidates per example (from
                   gold length, your own predictor, EOS-guess, whatever).
    memory_expand: callable(ell) -> memory already repeated to [B*ell, ...]
                   (mirror fairseq's duplicate_encoder_out).
    Picks the candidate with the best average log-prob (paper Sec 3.3).
    """
    B, ell = cand_lengths.size()
    flat_lengths = cand_lengths.reshape(B * ell)
    memory = memory_expand(ell)
    tokens, score = mask_predict_decode(decoder_forward, memory, flat_lengths,
                                        mask_id, pad_id, iterations)
    Lmax = tokens.size(1)
    tokens = tokens.view(B, ell, Lmax)
    score = score.view(B, ell)
    best = score.argmax(dim=1)
    return torch.stack([tokens[b, best[b]] for b in range(B)], dim=0)


def _show(t, worst, tgt, pad_mask, id2tok):
    row = tgt[0]
    def s(i):
        if pad_mask[0, i]: return "_"
        w = "*" if (worst is not None and worst[0, i]) else " "
        tk = id2tok[int(row[i])] if id2tok else str(int(row[i]))
        return f"{tk}{w}"
    print(f"  t={t}: " + " ".join(s(i) for i in range(tgt.size(1)) if not pad_mask[0, i] or i == 0))


# ============================================================
# DEMO with a toy bidirectional "decoder" (no training, just mechanics)
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    vocab = ["<pad>", "<mask>", "the", "withdrawal", "of", "french", "combat",
             "troops", "was", "completed", "on", "november", "20th", "<eos>", "WRONG"]
    tok2id = {w: i for i, w in enumerate(vocab)}
    id2tok = {i: w for w, i in tok2id.items()}
    PAD, MASK, WRONG = tok2id["<pad>"], tok2id["<mask>"], tok2id["WRONG"]
    V = len(vocab)

    TRUE = [tok2id[w] for w in
            ["the","withdrawal","of","french","combat","troops","was",
             "completed","on","november","20th","<eos>"]]      # N = 12

    def toy_decoder(tgt, memory):
        """Fake context-dependent model: a slot is confident only once its
        neighbours are filled; with no context it emits a low-confidence WRONG
        token (mimics the multi-modality errors that later iterations fix)."""
        B, L = tgt.shape
        logits = torch.full((B, L, V), -6.0)
        for b in range(B):
            for i in range(L):
                if tgt[b, i] == PAD:
                    logits[b, i, PAD] = 10.0; continue
                filled = 0
                for j in (i - 1, i + 1):
                    if 0 <= j < L and tgt[b, j] not in (MASK, PAD):
                        filled += 1
                if filled == 0:
                    p, target = 0.40, WRONG              # no context -> wrong, unsure
                else:
                    p, target = 0.55 + 0.20 * filled, TRUE[i]   # context -> right, sure
                logits[b, i, target] = torch.log(torch.tensor(p * (V - 1) / (1 - p)))
        return logits

    lengths = torch.tensor([12])
    print("Target we hope to reach:", " ".join(id2tok[t] for t in TRUE), "\n")
    print("Mask-predict, T=6  ( * = re-masked this round ):")
    tokens, score = mask_predict_decode(toy_decoder, None, lengths, MASK, PAD,
                                        iterations=6, verbose=True, id2tok=id2tok)
    out = [id2tok[int(t)] for t in tokens[0] if int(t) != PAD]
    print("\nFinal:", " ".join(out))
    print(f"avg log-prob score: {score.item():.3f}")

    # quick shape / correctness checks
    print("\nsanity: schedule of num_mask for N=12, T=6 ->",
          [(12 * (1 - t/6)) // 1 for t in range(1, 6)])
