import torch
from torch import nn

from typing import Sequence

class MLPBlock(nn.Module):
    def __init__(self, dims: Sequence[int] = (1024, 1024),
                dropout_rate: float = 0.0):
        super(MLPBlock, self).__init__()
    
        self.dims = dims
        self.dropout_rate = dropout_rate
        self.act_fn = nn.SiLU()
        # self.act_fn = nn.LeakyReLU()
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            # self.layers.append(nn.BatchNorm1d(self.dims[i + 1]))
            self.layers.append(self.act_fn)
            self.layers.append(nn.Dropout(self.dropout_rate))

        # self.layers.append(nn.Linear(self.dims[-2], self.dims[-1]))

    def forward(self, x):
        if len(self.dims) == 0:
            return x
        z = x
        for layer in self.layers:
            z = layer(z)

        return z

class Flowv2(nn.Module):
     def __init__(self, 
                 input_dim: int,
                 cond_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout: float,
                 cond: str,
                 num_flow_steps: int = 2,
                 scale_factor: float = 1.0,
                 latent_flow: bool = False):

        super(Flowv2, self).__init__()
        
        self.num_flow_steps = num_flow_steps

        self.cond = cond
        self.dropout = dropout

        self.x_0_encoder = MLPBlock(dims=[input_dim, hidden_dim], dropout_rate=dropout)
        self.noise_encoder = MLPBlock(dims=[input_dim, hidden_dim], dropout_rate=dropout)
        self.time_encoder = MLPBlock(dims=[1, hidden_dim], dropout_rate=dropout)
        self.x_encoder = MLPBlock(dims=[input_dim, hidden_dim], dropout_rate=dropout)
        
        if self.cond == "no_control":
            flow_input_dim = hidden_dim + hidden_dim + cond_dim  # x_t, t, (pert_z)
        elif self.cond == "control_mean":
            flow_input_dim = hidden_dim + hidden_dim + cond_dim + hidden_dim # x_t, t, (pert_z), x_0
        elif self.cond == "noised_control_mean":
            flow_input_dim = hidden_dim + hidden_dim + cond_dim + hidden_dim + hidden_dim # x_t, t, (pert_z), x_0, noise
        else:
            raise ValueError(f"Invalid condition: {self.cond}")
        flow_output_dim = output_dim

        self.velocity_net = MLPBlock(dims=[flow_input_dim, hidden_dim, flow_output_dim], dropout_rate=dropout)
        self.scale_factor = 1.0
 
 
     def forward(self, x_t, t, conditions, x_0_instrinsic=None, noise=None):
        x_t = self.x_encoder(x_t)
        t = self.time_encoder(t)
        if x_0_instrinsic.size(1) > 0:
            x_0_instrinsic = self.x_0_encoder(x_0_instrinsic)
        if noise.size(1) > 0:
            noise = self.noise_encoder(noise)
        
        inp = torch.cat([x_t, t, conditions, x_0_instrinsic, noise], dim=-1)
        v_pred = self.velocity_net(inp)
        return v_pred
