import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderProjectorLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
    
    def forward(self, x):
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class MoELayer_SMEAR(nn.Module):
    """
    Compact, vectorized SMEAR for EncoderProjectorLinear experts (2-layer MLP).
    Minimal Python loops: only used to stack expert params once per forward.
    """
    def __init__(self, experts: nn.ModuleList, input_dim: int, load_balancing_weight: float = 0.2):
        super().__init__()
        assert isinstance(experts, (list, nn.ModuleList)) and len(experts) > 0
        self.experts = experts
        self.num_experts = len(experts)
        self.input_dim = input_dim
        self.load_balancing_weight = load_balancing_weight

        # Router generates frame-wise logits
        self.router = nn.Linear(input_dim, self.num_experts)

    def forward(self, x, mask: torch.Tensor = None):
        """
        x: [B, L, D_in]
        mask (optional): [B, L] (1 for valid, 0 for pad)
        Returns:
            out: [B, L, out_dim]
            load_balance_loss: scalar tensor
            info: dict with utterance_probs and router_prob_mean
        """
        B, L, D_in = x.shape
        E = self.num_experts
        device = x.device
        dtype = x.dtype

        # import pdb
        # pdb.set_trace()
        # --- 1) Framewise router probs ---
        router_logits = self.router(x)                    # [B, L, E]
        router_probs = F.softmax(router_logits, dim=-1)   # [B, L, E]

        # optional mask: zero out padding positions
        if mask is not None:
            mask = mask.to(device=device, dtype=dtype)   # [B, L]
            router_probs = router_probs * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        else:
            denom = float(L)

        # --- 2) Load balancing loss (mean over batch & time) ---
        router_prob_mean = router_probs.mean(dim=(0, 1))   # [E]
        ideal_prob = torch.ones_like(router_prob_mean, device=device) / E
        load_balance_loss = self.load_balancing_weight * F.mse_loss(router_prob_mean, ideal_prob)

        # --- 3) Utterance-level probs ---
        if mask is not None:
            utterance_probs = router_probs.sum(dim=1) / denom  # [B, E]
        else:
            utterance_probs = router_probs.mean(dim=1)         # [B, E]

        # --- 4) Stack expert parameters (single short loop/list-comp) ---
        # For EncoderProjectorLinear we need linear1.weight/bias and linear2.weight/bias
        W1 = torch.stack([e.linear1.weight for e in self.experts], dim=0)  # [E, H, D_in]
        b1 = torch.stack([e.linear1.bias   for e in self.experts], dim=0)  # [E, H]
        W2 = torch.stack([e.linear2.weight for e in self.experts], dim=0)  # [E, out_dim, H]
        b2 = torch.stack([e.linear2.bias   for e in self.experts], dim=0)  # [E, out_dim]

        # --- 5) Merge per-utterance (einsum) - fully vectorized ---
        # merged_W1: [B, H, D_in], merged_b1: [B, H]
        merged_W1 = torch.einsum('be,ehi->bhi', utterance_probs, W1)
        merged_b1 = torch.einsum('be,eh->bh', utterance_probs, b1)
        # merged_W2: [B, out, H], merged_b2: [B, out]
        merged_W2 = torch.einsum('be,eoi->boi', utterance_probs, W2)
        merged_b2 = torch.einsum('be,eo->bo', utterance_probs, b2)

        # --- 6) Apply merged MLP in batch (vectorized) ---
        # first layer: x [B,L,D_in] @ merged_W1_T [B, D_in, H] -> [B, L, H]
        merged_W1_T = merged_W1.transpose(1, 2)    # [B, D_in, H]
        hidden = torch.einsum('bld,bdh->blh', x, merged_W1_T) + merged_b1.unsqueeze(1)
        hidden = F.relu(hidden)

        # second layer: hidden [B,L,H] @ merged_W2_T [B, H, out] -> [B, L, out]
        merged_W2_T = merged_W2.transpose(1, 2)    # [B, H, out]
        out = torch.einsum('blh,bho->blo', hidden, merged_W2_T) + merged_b2.unsqueeze(1)

        info = {
            'utterance_probs': utterance_probs.detach(),
            'router_prob_mean': router_prob_mean.detach()
        }
        return out, load_balance_loss, info


# ----------------- tiny test -----------------
if __name__ == "__main__":
    class Cfg:
        encoder_dim = 32
        llm_dim = 64
    cfg = Cfg()

    E = 4
    experts = nn.ModuleList([EncoderProjectorLinear(cfg) for _ in range(E)])
    layer = MoELayer_SMEAR_Compact(experts, input_dim=cfg.encoder_dim, load_balancing_weight=0.1)

    B, L = 3, 16
    x = torch.randn(B, L, cfg.encoder_dim)
    out, lb_loss, info = layer(x)

    print("out.shape", out.shape)                           # [B, L, llm_dim]
    print("load-balance loss", lb_loss.item())
    print("utterance probs shape", info['utterance_probs'].shape)  # [B, E]
