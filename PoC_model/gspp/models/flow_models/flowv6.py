import torch
from torch import nn
from typing import Sequence, Optional

# -------------------------
# Conditioning utils
# -------------------------
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    类似 Transformer 位置编码，让模型对时间 t 更敏感
    timesteps: [B] or [B,1]
    """
    if timesteps.dim() == 2 and timesteps.size(-1) == 1:
        timesteps = timesteps.squeeze(-1)
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=timesteps.device, dtype=torch.float32))
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # [B, dim]

# -------------------------
# Base MLP encoder block (unchanged)
# -------------------------
class MLPBlock(nn.Module):
    def __init__(
        self,
        dims: Sequence[int] = (1024, 1024),
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.dims = dims
        self.act_fn = nn.LeakyReLU()
        self.layers = nn.ModuleList()

        for i in range(len(self.dims) - 2):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if use_layer_norm:
                self.layers.append(nn.LayerNorm(self.dims[i + 1]))
            self.layers.append(self.act_fn)
            self.layers.append(nn.Dropout(dropout_rate))

        if len(self.dims) >= 2:
            self.layers.append(nn.Linear(self.dims[-2], self.dims[-1]))

    def forward(self, x):
        if len(self.dims) == 0:
            return x
        for layer in self.layers:
            x = layer(x)
        return x

# -------------------------
# AdaLN + conditional residual blocks
# -------------------------
class AdaLN(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 256, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(feat_dim, eps=eps, elementwise_affine=False)
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * feat_dim),
        )
        # AdaLN-Zero init (stable)
        last = self.to_gamma_beta[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, h, cond):
        h = self.ln(h)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

class CondResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.adaln1 = AdaLN(cond_dim, dim, hidden)
        self.fc1 = nn.Linear(dim, dim)
        self.adaln2 = AdaLN(cond_dim, dim, hidden)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h, cond):
        x = self.adaln1(h, cond)
        x = self.act(x)
        x = self.drop(self.fc1(x))
        x = self.adaln2(x, cond)
        x = self.act(x)
        x = self.drop(self.fc2(x))
        return h + x

class VelocityNetAdaLN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        width: int,
        out_dim: int,
        cond_dim: int,
        depth: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(
            [CondResBlock(width, cond_dim, hidden=512, dropout=dropout) for _ in range(depth)]
        )
        self.out = nn.Linear(width, out_dim)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.act(self.inp(x))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)

# -------------------------
# Flowv6 
# -------------------------
class Flowv6(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,  # pert output dim
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        cond: str,
        num_flow_steps: int = 50,
        scale_factor: float = 1.0,
        latent_flow: bool = False,
        # new knobs
        velocity_depth: int = 6,
        adaln_use_time: bool = False,
        use_global_norm: bool = True,
    ):
        super(Flowv6, self).__init__()

        self.cond = cond
        self.hidden_dim = hidden_dim
        self.adaln_use_time = adaln_use_time
        self.use_global_norm = use_global_norm

        latent_dim = hidden_dim // 8

        # Encoders (keep your design)
        self.x_encoder = MLPBlock(
            dims=[input_dim, hidden_dim, latent_dim], dropout_rate=dropout, use_layer_norm=True
        )
        self.time_encoder = MLPBlock(
            dims=[hidden_dim, hidden_dim, latent_dim], dropout_rate=dropout, use_layer_norm=True
        )
        self.x_0_encoder = MLPBlock(
            dims=[input_dim, hidden_dim, latent_dim], dropout_rate=dropout, use_layer_norm=True
        )
        self.noise_encoder = MLPBlock(
            dims=[input_dim, hidden_dim, latent_dim], dropout_rate=dropout, use_layer_norm=True
        )

        base_dim = latent_dim + latent_dim + cond_dim
        if self.cond == "no_control":
            self.flow_input_dim = base_dim
        elif self.cond == "control_mean":
            self.flow_input_dim = base_dim + latent_dim
        else:
            raise ValueError(f"Invalid condition: {self.cond}")

        # Per-branch norms (keep)
        self.x_t_norm = nn.LayerNorm(latent_dim)
        self.t_norm = nn.LayerNorm(latent_dim)
        self.conditions_norm = nn.LayerNorm(cond_dim)
        self.x_0_norm = nn.LayerNorm(latent_dim)

        # Global norm (optional)
        self.global_norm = nn.LayerNorm(self.flow_input_dim)

        # Decide cond dim for AdaLN
        self.cond_dim = cond_dim
        self.cond_for_adaln_dim = cond_dim + (latent_dim if adaln_use_time else 0)

        # Replace velocity_net with AdaLN-conditioned residual network
        self.velocity_net = VelocityNetAdaLN(
            in_dim=self.flow_input_dim,
            width=hidden_dim,
            out_dim=output_dim,
            cond_dim=self.cond_for_adaln_dim,
            depth=velocity_depth,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditions: torch.Tensor,
        x_0_instrinsic: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ):
        # Encode branches
        x_t_enc = self.x_t_norm(self.x_encoder(x_t))  # [B, latent_dim]

        # NOTE: your original uses t*1000; keep it
        t_emb = timestep_embedding(t * 1000, self.hidden_dim)  # [B, hidden_dim]
        t_enc = self.t_norm(self.time_encoder(t_emb))          # [B, latent_dim]

        conditions_normed = self.conditions_norm(conditions)   # [B, cond_dim]

        feat_list = [x_t_enc, t_enc, conditions_normed]

        if self.cond in ["control_mean"] and x_0_instrinsic is not None:
            x_0_enc = self.x_0_norm(self.x_0_encoder(x_0_instrinsic))  # [B, latent_dim]
            feat_list.append(x_0_enc)

        combined = torch.cat(feat_list, dim=-1)  # [B, flow_input_dim]
        if self.use_global_norm:
            combined = self.global_norm(combined)

        # Cond used for AdaLN modulation
        if self.adaln_use_time:
            cond_for_adaln = torch.cat([conditions_normed, t_enc], dim=-1)  # [B, cond_dim + latent_dim]
        else:
            cond_for_adaln = conditions_normed  # [B, cond_dim]

        v_pred = self.velocity_net(combined, cond_for_adaln)
        return v_pred
