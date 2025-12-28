from typing import Any, Dict

import torch
from torch import hinge_embedding_loss, nn
from torch.nn import functional as F

from torch_geometric.nn import MLP

from gspp.data.graphmodule import GSPGraph

from gspp.constants import CONTROL_LABEL

from gspp.models.basal_state_models.vae import VAE
from gspp.models.basal_state_models.mlp import BasalMLP
from gspp.models.basal_state_models.moe import MoE, MoA
from gspp.models.pert_models.basic_gnn import GNN
from gspp.models.pert_models.hybrid_wrapper import HybridWrapper
from gspp.models.pert_models.multi_graph import MultiGraph
from gspp.models.pert_models.exphormer import ExphormerModel
from gspp.models.flow_models.flowv1 import Flowv1
from gspp.models.flow_models.flowv2 import Flowv2
from gspp.models.flow_models.flowv3 import Flowv3
from gspp.models.flow_models.flowv4 import Flowv4
from gspp.models.flow_models.flowv5 import Flowv5
from gspp.models.flow_models.flowv6 import Flowv6
from collections.abc import Callable, Sequence
from tqdm import tqdm

BASAL_STATE_MODEL_DICT = {
    "vae": VAE,
    "mlp": BasalMLP,
    "moe": MoE,
    "moa": MoA,
}

PERT_MODEL_DICT = {
    "mlp": MLP,
    "gnn": GNN,
    "hybrid_wrapper": HybridWrapper,
    "multilayer": MultiGraph,
    "exphormer": ExphormerModel,
}

FLOW_MODEL_DICT = {
    "v1": Flowv1,
    "v2": Flowv2,
    "v3": Flowv3,
    "v4": Flowv4,
    "v5": Flowv5,
    "v6": Flowv6,
}


class TxPertFlow(nn.Module):
    """
    Flow Matching version of PertPredictor.

    思路：
    - x0 = control expression (intrinsic)
    - x1 = perturbed expression (target)
    - 训练：对每个样本随机采 t ∈ [0, 1]，构造 x_t = (1 - t) * x0 + t * x1
      真值速度 v* = x1 - x0
      模型预测 v_theta(x_t, t, cond) 拟合 v*
    - 推断：从 x(0) = x0 出发，用 ODE: dx/dt = v_theta(x, t, cond) 积分到 t=1 得到预测 x1_hat
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        p_emb_dim: int,
        adata_output_dim: int,
        graph: GSPGraph,
        cntr_model_args: Dict[str, Any],
        pert_model_args: Dict[str, Any],
        flow_model_args: Dict[str, Any],
        hidden_dim: int = 512,
        latent_dim: int = 64,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.2,
        omit_cntr: bool = False,
        no_basal_model: bool = False,
        bce_loss: bool = False,
        pert_input_dim: int = None,
        mse_weight: float = 1.0,
    ):
        super(TxPertFlow, self).__init__()

        self.mse_weight = mse_weight
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p_emb_dim = p_emb_dim
        self.no_basal_model = no_basal_model
        self.bce_loss = bce_loss

        self.adata_output_dim = adata_output_dim        
        self.cntr_model = None
        
        # Perturbation model
        self.pert_model_type = pert_model_args.pop("model_type")
        if self.pert_model_type == 'None':
            self.pert_model = None
            self.cond_dim = 0
            print("=========pert_model is None=========")
        else:
            self.pert_model = PERT_MODEL_DICT[self.pert_model_type](
                out_dim=latent_dim,
                graph=graph,
                **pert_model_args,
            )
            self.omit_cntr = omit_cntr
            self.cond_dim = self.p_emb_dim + latent_dim
            # Graph
            edge_index, edge_weight, _ = next(iter(graph.graph_dict.values()))
            self.edge_index = torch.Tensor(edge_index)
            self.edge_weight = torch.Tensor(edge_weight)
            print("=========pert_model=========", self.pert_model)

        # Flow Matching model
        self.flow_model_type = flow_model_args.pop("model_type")
        self.num_flow_steps = flow_model_args.pop("num_flow_steps")
        print("=========num_flow_steps=========", self.num_flow_steps)
        self.latent_flow = flow_model_args.pop("latent_flow")
        self.scale_factor = flow_model_args.pop("scale_factor", 1.0)
        self.cond = flow_model_args.pop("cond")
        self.network_type = flow_model_args.pop("network_type")
        self.noise_scale = flow_model_args.pop("noise_scale")

        self.flow_model = FLOW_MODEL_DICT[self.network_type](
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            cond_dim=self.cond_dim,
            output_dim=output_dim,
            dropout=dropout,
            cond=self.cond,
            num_flow_steps=self.num_flow_steps,
            scale_factor=self.scale_factor,
            latent_flow=self.latent_flow,
        )
        print("=========flow model=========", self.flow_model)

    # ======================
    # Perturbation model 核心部分
    # ======================

    def compute_pert_z(self, cntr, pert_idxs):
        batch_size = cntr.size(0)

        self.edge_index = self.edge_index.to(cntr.device)
        self.edge_weight = self.edge_weight.to(cntr.device)

        z_p = self.pert_model.forward(
            self.edge_index, self.edge_weight
        )

        pert_z = torch.zeros(batch_size, z_p.size(-1), device=cntr.device)

        for i, perts in enumerate(pert_idxs):
            for p in perts:
                if self.omit_cntr and p in [-1, CONTROL_LABEL]:
                    continue

                if self.pert_model.use_cntr or self.pert_model.cat_cntr:
                    pert_z[i] = pert_z[i] + z_p[p][i]
                else:
                    # print(z_p.shape, p)
                    if p == 'ctrl':
                        p = -1
                    pert_z[i] = pert_z[i] + z_p[p]        

        return pert_z
    
    def flow_matching_loss(self, prediction, v_true):
        """
        cntr: [B, D] control intrinsic
        target: [B, D] target expression
        intrinsic_mean: [B, D] intrinsic mean
        intrinsic_log_var: [B, D] intrinsic log variance
        """

        # 计算损失：MSE(v_pred, v_true)
        loss = F.mse_loss(prediction, v_true)
        
        # 2. 方向损失 (你现在已经做得很好，可以保留或略微减小权重)
        cos_sim = F.cosine_similarity(prediction, v_true, dim=-1).mean()
        cos_loss = 1 - cos_sim
        
        # 3. 幅度损失 (强制模长对齐)
        # 衡量预测向量的长度和真实长度的差距
        norm_loss = F.mse_loss(prediction.norm(dim=-1), v_true.norm(dim=-1))
        total_loss = loss + 0* cos_loss + 0 * norm_loss # 0.1 is the weight

        logs = {
            "flow_loss": loss,
            "cos_loss": cos_loss,
            "norm_loss": norm_loss,
            "v_target_norm": v_true.norm(dim=-1).mean(),
            "v_pred_norm": prediction.norm(dim=-1).mean(),
            "v_target_mean": v_true.mean(),
            "v_pred_mean": prediction.mean(),
            "v_cosine_sim": F.cosine_similarity(prediction, v_true, dim=-1).mean()
        }

        return total_loss, logs




    def forward(self, cntr, pert_idxs, p_emb, target):
            # ... (前面的 conditions 准备逻辑保持不变)

            # 1. 决定画布起点 x_0
            if self.flow_model_type == "random_noise":
                # print("=========flow_model_type is random_noise=========")
                # 关键：匹配数据 Std 0.4。如果是全新的生成，这会让路径非常平滑
                x_0 = torch.randn_like(cntr) * self.noise_scale #* 0.05 0.1 0.2 0.4 0.6 1.0
            elif self.flow_model_type == "control_mean":
                x_0 = cntr
            
            x_1 = target # 终点是真实的被扰动细胞

            # 2. 采样时间 t
            B = x_0.size(0)
            t = torch.rand(B, 1, device=cntr.device)

            # 3. 构造中间态 x_t (Flow Matching 线性插值)
            x_t = (1 - t) * x_0 + t * x_1
            
            # 4. 准备参考图 (Condition)
            # 即使起点是 noise，我们也把 cntr 喂给模型作为“参考图”
            x_0_instrinsic = cntr if self.cond != "no_control" else torch.empty(B, 0).to(cntr.device)
            
            # 如果需要给参考图也加点噪（增加鲁棒性）
            noise_ref = torch.randn_like(cntr) * 0.6 if self.cond == "noised_control_mean" else torch.empty(B, 0).to(cntr.device)
            
            
            if self.pert_model is not None:
                pert_z = self.compute_pert_z(cntr, pert_idxs)
                conditions = pert_z
                if self.p_emb_dim > 0:
                    conditions = torch.cat([conditions, p_emb], dim=-1)
            else:
                conditions = torch.zeros(cntr.size(0), 0, device=cntr.device)


            # 5. 预测速度
            v_pred = self.flow_model(x_t, t, conditions, x_0_instrinsic, noise_ref)

            # 6. 计算真值速度
            v_true = x_1 - x_0
            # v_true = x_1
            
            return v_pred, v_true



    @torch.no_grad()
    def _ode_solve(self, cntr, p_emb, pert_idxs):
        self.eval()
        B = cntr.size(0)
        device = cntr.device

        # 1. 准备 Conditions (与 forward 保持一致)
        if self.pert_model is not None:
            pert_z = self.compute_pert_z(cntr, pert_idxs)
            conditions = pert_z
            if self.p_emb_dim > 0:
                conditions = torch.cat([conditions, p_emb], dim=-1)
        else:
            conditions = torch.zeros(B, 0, device=device)

        # 2. 初始化起点 x_t
        if self.flow_model_type == "random_noise":
            # 这里的系数必须与训练时的 forward 严格一致！
            x_t = torch.randn_like(cntr) * self.noise_scale #* 0.05 0.1 0.2 0.4 0.6 1.0
        elif self.flow_model_type == "control_mean":
            x_t = cntr
        else:
            raise ValueError(f"Invalid flow model type: {self.flow_model_type}")

        # 3. 准备参考图 (Intrinsic Condition)
        # 注意：这里的 noise 也要采样，如果是 "noised_control_mean" 模式
        if self.cond == "control_mean":
            x_0_intrinsic = cntr
            noise_ref = torch.zeros(B, 0, device=device)
        elif self.cond == "no_control":
            x_0_intrinsic = torch.zeros(B, 0, device=device)
            noise_ref = torch.zeros(B, 0, device=device)

        # 4. Euler 积分求解
        t0, t1 = 0.0, 1.0
        # 建议：如果训练效果好但生成不够逼真，可以把 num_flow_steps 设大一点（如 50）
        steps = self.num_flow_steps
        dt = (t1 - t0) / steps

        for i in range(steps):
            # t 从 0 逐渐增加到 1-dt
            curr_t = t0 + i * dt
            t_vec = torch.full((B, 1), curr_t, device=device)
            
            # 预测当前位置的速度
            v = self.flow_model.forward(x_t, t_vec, conditions, x_0_intrinsic, noise_ref)
            
            # 更新位置：x(t + dt) = x(t) + v * dt
            x_t = x_t + v * dt
        
        return x_t # 最终到达 t=1，即预测的 perturbed cell


    def create_supra_adj(self, graphs):
        """
        Create supra-adjacency matrix for multilayer GNN.
        """
        device = graphs[0][0].device
        nodes_per_layer = [g[2] for g in graphs]
        total_nodes = sum(nodes_per_layer)

        edges_list = []
        weights_list = []
        offset = 0

        for i, (edge_index, edge_weight, num_nodes) in enumerate(graphs):
            # Check edge index bounds before offset
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                raise ValueError(
                    f"Layer {i}: Edge index {max_idx} exceeds number of nodes {num_nodes}"
                )

            # Create new edge index with offset
            new_edge_index = edge_index.clone()
            new_edge_index += offset

            # Validate after offset
            new_max_idx = new_edge_index.max().item()
            if new_max_idx >= total_nodes:
                raise ValueError(
                    f"Layer {i}: Offset edge index {new_max_idx} exceeds total nodes {total_nodes}"
                )

            edges_list.append(new_edge_index)
            weights_list.append(edge_weight)

            offset += num_nodes

        # Combine edges and weights
        supra_edge_index = torch.cat(edges_list, dim=1)
        supra_edge_weight = torch.cat(weights_list)

        return supra_edge_index, supra_edge_weight, total_nodes