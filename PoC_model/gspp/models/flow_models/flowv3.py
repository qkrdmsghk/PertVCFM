import torch
from torch import nn

from typing import Sequence
import math

# --- 1. 标准的时间位置编码 (必须有！) ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [B, 1]
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# --- 2. 带残差连接的 MLP Block ---
class ResMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, t_emb=None):
        # 如果有 Time Embedding，简单地加在 Feature 上 (或者做 AdaLN)
        h = x
        if t_emb is not None:
            h = h + t_emb
            
        return x + self.fc(self.act(self.norm(h))) # Residual: x + f(x)

class Flowv3(nn.Module):
     def __init__(self, 
                 input_dim: int,
                 cond_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout: float,
                 cond: str,
                 num_flow_steps: int = 2,
                 scale_factor: float = 1.0,
                 latent_flow: bool = False,
                 num_layers: int = 2):

        super(Flowv3, self).__init__()
        
        self.cond = cond
        self.input_dim = input_dim
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 2. Input Projection (先把所有条件拼起来，统一映射一次)
        # 计算拼接后的维度

        concat_dim = input_dim # x_t
        concat_dim += cond_dim # + conditions (perturbation embeddings)

        if cond == "no_control":
            pass
        elif cond == "control_mean":
            concat_dim += input_dim # + x_0 (latent control)
        elif cond == "noised_control_mean":
            concat_dim += input_dim + input_dim # + x_0 (latent control) + noise
        else:
            raise ValueError(f"Invalid condition: {cond}")
        
        self.input_proj = nn.Linear(concat_dim, hidden_dim)
        # 3. Main Backbone (ResNet)
        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # 4. Final Output Projection
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.SiLU(),
            # nn.Dropout(dropout)
        )
         
 
     def forward(self, x_t, t, conditions, x_0_instrinsic=None, noise=None):
        """
        x_t: [B, D]
        t: [B, 1]
        conditions: [B, C] (Perturbation Embeddings)
        x_0_intrinsic: [B, D] (Latent Control)
        """
        
        # 1. Process Time
        t_emb = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Concatenate Inputs
        inp = torch.cat([x_t, conditions, x_0_instrinsic, noise], dim=-1)
        
        # 3. Project Inputs
        inp = self.input_proj(inp) # [B, hidden_dim]
        
        # 4. Pass through ResNet Blocks 
        for block in self.blocks:
            inp = block(inp, t_emb)
            
        # 5. Output Velocity v_pred
        v_pred = self.final_layer(inp) # [B, output_dim]
        
        return v_pred
