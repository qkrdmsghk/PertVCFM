from typing import Any, Dict

import torch

import numpy as np
import torch.nn as nn
from lightning import LightningModule
from torch.utils.data import DataLoader

from gspp.data.graphmodule import GSPGraph
from gspp.models.txpertflow import TxPertFlow
from gspp.models.baselines import MeanBaseline, ExperimentalAccuracy
from gspp.evaluation import evaluate
from gspp.metrics import log_metrics, compute_metrics

import gspp.constants as cs

MODEL_DICT = {
    "txpertflow": TxPertFlow,
    "mean_baseline": MeanBaseline,
    "experimental_baseline": ExperimentalAccuracy,
}


class FlowPertPredictor(LightningModule):
    """
    Predictor class that manages training, testing, logging and instantiate the model. To allow for different model types,
    the model is selected based on the `model_type` argument in the `model_args` dictionary and the loss function is defined
    in the model class rather than here.

    Args:
        input_dim (int): The input dimension of the data.
        output_dim (int): The output dimension of the data.
        adata_output_dim (int): The output dimension of the adata.
        model_args (Dict[str, Any]): The arguments for the model.
        graph (GSPGraph): Graph object containing the edge index and edge weight.
        p_emb_dim (int): The input dimension of the perturbation embeddings.
        no_pert (bool): Whether to use perturbations.
        lr (float): The learning rate.
        min_lr (float): Minimum learning rate for scheduler
        weight_decay (float): The weight decay.
        device (str): The device.
        run_val_on_train_data (bool): Whether to run validation on the train data.
        match_cntr_for_eval (bool): Whether to match controls for evaluation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        p_emb_dim: int,
        adata_output_dim: int,
        model_args: Dict[str, Any],
        graph: GSPGraph = None,
        pert_input_dim: int = None,
        min_lr: float = 1e-6,
        lr_scheduler_args: Dict[str, Any] = {},
        weight_decay: float = 0.0,
        # device: str = "cpu",
        run_val_on_train_data: bool = False,
        match_cntr_for_eval: bool = True,
        ae_model: nn.Module = None,
    ):
        super(FlowPertPredictor, self).__init__()

        self.save_hyperparameters(ignore=["graph", "ae_model"])
        self.ae_model = ae_model
        if self.ae_model is not None:
            print(f"✅ [FlowPertPredictor] Pretrained AE detected. Switching to LATENT SPACE Flow Matching.")
            self.ae_model.eval()
            self.ae_model.requires_grad_(False) # Freeze AE weights            
            # Extract latent dimension (assumed to be 64 or defined in config)
            latent_dim = 512
            # OVERRIDE dimensions: The internal Flow Model sees Z, not X
            # So input_dim becomes 64, not 4096
            input_dim = latent_dim
            output_dim = latent_dim
            adata_output_dim = latent_dim 
            print(f"ℹ️  [FlowPertPredictor] Internal dimensions overridden to: {latent_dim}")
        # ------------------------------------------------
        # self.slow_benchmark = model_args['slow_benchmark']
        self.run_val_on_train_data = run_val_on_train_data
        self.match_cntr_for_eval = match_cntr_for_eval
        self.min_lr = min_lr
        self.lr_scheduler_args = lr_scheduler_args
        self.weight_decay = weight_decay

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adata_output_dim = adata_output_dim

        print("=========model_args=========", model_args)
        model_args.pop('no_pert_model')
        model_args.pop('slow_benchmark')
        self.model_type = model_args.pop('model_type')
        self.lr = model_args.pop('lr')

        cntr_model_args = model_args.pop('cntr_model')
        pert_model_args = model_args.pop('pert_model')
        flow_model_args = model_args.pop('flow_model')
        self.basal_mode = model_args.pop("basal_mode", "construct_control")

        self.model = MODEL_DICT[self.model_type](
            input_dim=input_dim,
            output_dim=output_dim,
            p_emb_dim=p_emb_dim,
            adata_output_dim=adata_output_dim,
            graph=graph,
            cntr_model_args=cntr_model_args,
            pert_model_args=pert_model_args,
            flow_model_args=flow_model_args,
            pert_input_dim=pert_input_dim,
            **model_args,
        )

    def forward(self, cntr, pert_idxs, p_emb, target):
        device = next(self.parameters()).device
        cntr = cntr.to(device)
        p_emb = p_emb.to(device)
        target = target.to(device)

        return self.model.forward(cntr, pert_idxs, p_emb, target)

    def loss(self, prediction, v_true):
        # Call loss function provided through the model
        loss, logs = self.model.flow_matching_loss(prediction, v_true)
    
        self.log_dict(
            logs,
            prog_bar=True,
            logger=True,
        )
        return loss, logs

    def training_step(self, batch, batch_idx):
        
        # --- [2] 训练时的编码逻辑 (Encode Logic) ---
        raw_cntr = batch.control
        raw_target = batch.x # 假设原始数据维度是 input_dim (需确保这里切片正确)
        
        if self.ae_model is not None:
            # Latent Mode: 先把 Raw Data 压成 Z
            with torch.no_grad(): # AE 不参与梯度计算
                curr_cntr = self.ae_model.encode(raw_cntr)
                curr_target = self.ae_model.encode(raw_target)
        else:
            # Raw Mode: 直接用
            curr_cntr = raw_cntr
            curr_target = raw_target

        # 前向传播 (在 Latent Space 或 Raw Space)
        prediction, v_true = self.forward(
            cntr=curr_cntr,
            pert_idxs=batch.pert_idxs,
            p_emb=batch.p,
            target=curr_target
        )

        loss, logs = self.loss(prediction, v_true)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], prog_bar=False, sync_dist=True)

        for k, v in logs.items():
            self.log(f"train/{k}", v, prog_bar=False, sync_dist=True)
            
        return loss

    def sample_inference(self, cntr, pert_idxs, p_emb, cell_types):

        """
        推理逻辑：
        1. 如果有 AE: Encode(x) -> ODE Solve(z) -> Decode(z_hat) -> x_hat
        2. 如果无 AE: ODE Solve(x) -> x_hat
        """
        device = next(self.parameters()).device
        cntr = cntr.to(device)
        p_emb = p_emb.to(device)

        # --- [3] 推理时的三明治结构 ---
        if self.ae_model is not None:
            # Step 1: Encode Control -> Z_control
            with torch.no_grad():
                z_cntr = self.ae_model.encode(cntr)
            
            # Step 2: Flow Matching ODE Solver (in Latent Space)
            # 输出的是 z_perturb_pred
            z_pred = self.model._ode_solve(z_cntr, p_emb, pert_idxs)
            
            # Step 3: Decode Z_perturb_pred -> X_perturb_pred
            with torch.no_grad():
                prediction = self.ae_model.decode(z_pred)
        else:
            # 原始逻辑
            prediction = self.model._ode_solve(cntr, p_emb, pert_idxs)

        return prediction

    def general_validation_epoch(
        self, stage: str, loader: DataLoader, metric_stage: str = cs.FAST
    ):
        """
        Run a general validation epoch. This function is used to run the validation on the train, val, and test data.

        Args:
            stage (str): The stage of the experiment (train, val, test).
            loader (DataLoader): The data loader.
            metric_stage (str): The metric stage (fast, extended).
        """
        test_adata = self.trainer.datamodule.adata

        if "baseline" in self.model_type:
            results = self.model.apply_baseline(self.trainer.datamodule.test_data)
        else:
            results = evaluate(
                loader,
                self,
                self.device,
                test_adata,
                self.trainer.datamodule.id2pert,
            )

        metrics, test_pert_res = compute_metrics(
            results, test_adata, metric_stage, match_cntr=self.match_cntr_for_eval
        )

        log_metrics(
            metrics, test_pert_res, stage, self.trainer.datamodule.subgroup, self
        )

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        if self.run_val_on_train_data:
            self.general_validation_epoch(
                cs.TRAIN, self.trainer.datamodule.train_dataloader()
            )
        self.general_validation_epoch(cs.VAL, self.trainer.datamodule.val_dataloader())
        self.general_validation_epoch(cs.TEST, self.trainer.datamodule.test_dataloader())

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        # metric_set = cs.SLOW if self.slow_benchmark else cs.EXTENDED
        metric_set = cs.FAST
        self.general_validation_epoch(
            cs.TEST, self.trainer.datamodule.test_dataloader(), metric_set
        )

    def configure_optimizers(self):
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=self.lr, weight_decay=self.weight_decay
        )

        schedulers = []
        milestones = []

        warmup_epochs = self.lr_scheduler_args.get("warmup_epochs", 0)
        if warmup_epochs > 0:
            print("=========warmup_epochs=========", warmup_epochs)
            schedulers.append(
                cs.SCHEDULER_DICT["linear"](
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
            )
            milestones.append(warmup_epochs)

        scheduler_type = self.lr_scheduler_args.get("type", None)
        if scheduler_type is not None:
            total_epochs = self.lr_scheduler_args["total_epochs"]
            schedulers.append(
                cs.SCHEDULER_DICT[scheduler_type](
                    optimizer, T_max=total_epochs - warmup_epochs, eta_min=self.lr_scheduler_args["min_lr"]
                )
            )

        scheduler = (
            cs.SCHEDULER_DICT["cosine"](
                optimizer, schedulers, milestones=milestones
            )
            if len(schedulers) > 0
            else None
        )

        return (
            optimizer
            if scheduler is None
            else {"optimizer": optimizer, "lr_scheduler": scheduler}
        )
