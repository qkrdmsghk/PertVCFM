import torch
from torch import nn

from typing import Sequence

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    类似于 Transformer 的位置编码，让模型对时间 t 更敏感
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class MLPBlock(nn.Module):
    def __init__(self, dims: Sequence[int] = (1024, 1024),
                dropout_rate: float = 0.0,
                use_layer_norm: bool = False):
        super(MLPBlock, self).__init__()
    
        self.dims = dims
        self.act_fn = nn.LeakyReLU()
        self.layers = nn.ModuleList()
        
        # 遍历到倒数第二层
        for i in range(len(self.dims) - 2):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if use_layer_norm:
                self.layers.append(nn.LayerNorm(self.dims[i + 1]))
            self.layers.append(self.act_fn)
            self.layers.append(nn.Dropout(dropout_rate))

        # 最后一层：只给 Linear，不加任何约束
        if len(self.dims) >= 2:
            self.layers.append(nn.Linear(self.dims[-2], self.dims[-1]))

    def forward(self, x):
        if len(self.dims) == 0:
            return x
        for layer in self.layers:
            x = layer(x)
        return x

class Flowv5(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 cond_dim: int, # pert output dim
                 hidden_dim: int, 
                 output_dim: int, 
                 dropout: float,
                 cond: str,
                 num_flow_steps: int = 50, # 建议生成时步数多一点
                 scale_factor: float = 1.0,
                 latent_flow: bool = False):

        super(Flowv5, self).__init__()
        
        self.cond = cond
        self.hidden_dim = hidden_dim
        latent_dim = hidden_dim//8

        # 各分支 Encoder
        self.x_encoder = MLPBlock(dims=[input_dim, hidden_dim, latent_dim], dropout_rate=dropout, use_layer_norm=True)
        self.time_encoder = MLPBlock(dims=[hidden_dim, hidden_dim, latent_dim], dropout_rate=dropout, use_layer_norm=True)
        self.x_0_encoder = MLPBlock(dims=[input_dim, hidden_dim, latent_dim], dropout_rate=dropout, use_layer_norm=True)
        self.noise_encoder = MLPBlock(dims=[input_dim, hidden_dim, latent_dim], dropout_rate=dropout, use_layer_norm=True)
        
        # 严格计算输入维度
        # 基础部分：x_t 和 t 的 embedding + 外部传入的 conditions (pert_z + p_emb)
        base_dim = latent_dim + latent_dim + cond_dim
        
        if self.cond == "no_control":
            self.flow_input_dim = base_dim
        elif self.cond == "control_mean":
            self.flow_input_dim = base_dim + latent_dim
        else:
            raise ValueError(f"Invalid condition: {self.cond}")

        self.x_t_norm = nn.LayerNorm(latent_dim)
        self.t_norm = nn.LayerNorm(latent_dim)
        self.conditions_norm = nn.LayerNorm(cond_dim)
        self.x_0_norm = nn.LayerNorm(latent_dim)
        
        # 核心：在拼接后加入全局 LayerNorm，这是解决震荡的“特效药”
        self.global_norm = nn.LayerNorm(self.flow_input_dim)

        # 速度预测网络
        self.velocity_net = MLPBlock(dims=[self.flow_input_dim, hidden_dim, output_dim], dropout_rate=dropout, use_layer_norm=True)

    def forward(self, x_t, t, conditions, x_0_instrinsic=None, noise=None):
        # 1. 对各个输入进行编码
        x_t_enc = self.x_t_norm(self.x_encoder(x_t))
        t_enc = self.t_norm(self.time_encoder(timestep_embedding(t*1000, self.hidden_dim)))
        conditions = self.conditions_norm(conditions)

        # 2. 构造特征列表（顺序必须与 __init__ 对应）
        # [x_t_enc, t_enc, conditions]
        feat_list = [x_t_enc, t_enc, conditions]
        
        # 3. 按条件注入 Control 信息
        if self.cond in ["control_mean"]:
            if x_0_instrinsic is not None:
                x_0_enc = self.x_0_norm(self.x_0_encoder(x_0_instrinsic))
                feat_list.append(x_0_enc)
        
        # 5. 拼接 + 归一化
        # 这样无论支路数值多大，进入 velocity_net 前都会被拉回到标准分布
        combined = torch.cat(feat_list, dim=-1)
        combined = self.global_norm(combined)
        
        # 6. 预测速度向量 v
        v_pred = self.velocity_net(combined)
        return v_pred