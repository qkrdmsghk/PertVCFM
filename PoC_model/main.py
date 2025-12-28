import sys
from tabnanny import verbose
import hydra

from omegaconf import DictConfig, OmegaConf

import torch

from time import time

# from loguru import logger
from lightning.pytorch.loggers import WandbLogger

import traceback
from lightning import Trainer

from gspp.predictor import PertPredictor

from gspp.data.graphmodule import GSPGraph
from gspp.data.datamodule import PertDataModule

from gspp.utils import set_seed
from gspp.constants import CHECKPOINT_DIR
from gspp.utils import download_files_from_zenodo
from lightning.pytorch.callbacks import ModelCheckpoint

from gspp.FlowPertPredictor import FlowPertPredictor

# Main function to initialize Hydra and parse the configuration
@hydra.main(config_name="config-gat", config_path="configs", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # download_files_from_zenodo(
    #     "15420279"
    # )
    try:
        t0 = time()
        infer(cfg)
        # logger.info(f"Time taken: {time() - t0} seconds")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def infer(cfg: DictConfig):
    # Check if GPU is available and force single device usage
    gpu_available = torch.cuda.is_available()
    
    # Force using a specific GPU device (e.g., cuda:0)
    if gpu_available:
        device = "cuda:0"  # Force single GPU
        torch.cuda.set_device(0)  # Set default CUDA device
    else:
        device = "cpu"

    torch.set_float32_matmul_precision('medium')

    set_seed(cfg.seed)

    data_args = OmegaConf.to_container(cfg.datamodule, resolve=True)

    datamodule = PertDataModule(**data_args)
    datamodule.prepare_data(seed=cfg.seed)

    
    if hasattr(cfg, "wandb") and cfg.wandb.name != "None" and cfg.mode =="baseline":
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            save_dir=cfg.wandb.save_dir,
            log_model=True,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.datamodule.task_type}/{cfg.wandb.name}_{wandb_logger.experiment.id}",
            filename="best-{epoch}-{step}",
            save_top_k=5,
            monitor='val_pearson_delta',
            mode='max',
            # every_n_train_steps=10000,
        )
    else:
        print("[DEBUG MODE] WandB disabled")
        wandb_logger = None
        checkpoint_callback = None



    trainer = Trainer(
        accelerator="gpu" if gpu_available else "cpu",
        devices=1,  # Force single device
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=2,
        logger=wandb_logger if hasattr(cfg, "wandb") else None, 
        num_sanity_val_steps=-1,
        callbacks=[checkpoint_callback] if checkpoint_callback is not None else None,
        gradient_clip_val=1.0,
    )

    if cfg.mode == "baseline":
        datamodule.setup("fit")
        if cfg.model.model_type == "txpert":
            # Instatiate graph(s) used by the GNN
            graph = GSPGraph(
                pert2id=datamodule.pert2id,
                gene2id=datamodule.gene2id,
                **cfg.graph,
            )
            GSP_model = PertPredictor(
                input_dim=datamodule.input_dim,
                output_dim=datamodule.output_dim,
                p_emb_dim=datamodule.p_emb_dim,
                adata_output_dim=datamodule.adata_output_dim,
                graph=graph,
                model_args=OmegaConf.to_container(cfg.model, resolve=True),
                match_cntr_for_eval=cfg.match_cntr_for_eval,
            )
            trainer.fit(GSP_model, datamodule)
        elif cfg.model.model_type == "txpertflow":
            ae_model = None
            print("🔄 No AE Checkpoint Detected. Using data space Flow Matching...")

            graph = GSPGraph(
                pert2id=datamodule.pert2id,
                gene2id=datamodule.gene2id,
                **cfg.graph,
            )
            GSP_Flow_model = FlowPertPredictor(
                input_dim=datamodule.input_dim,
                output_dim=datamodule.output_dim,
                p_emb_dim=datamodule.p_emb_dim,
                adata_output_dim=datamodule.adata_output_dim,
                graph=graph,
                model_args=OmegaConf.to_container(cfg.model, resolve=True),
                ae_model=ae_model,
            )
            trainer.fit(GSP_Flow_model, datamodule)

        
        else:
            raise ValueError(f"Invalid model type: {cfg.model.model_type}")
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")



if __name__ == "__main__":
    main()