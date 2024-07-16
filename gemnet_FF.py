import torch
import itertools
import json
import hashlib
import os
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter

import umap
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch_scatter
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR
import wandb
import numpy as np
from scipy.ndimage import convolve1d
from collections import Counter
from pymatgen.core import Structure
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch.utils.data import Dataset as DS  # collision
from torch_geometric.utils.convert import (
    to_scipy_sparse_matrix,
)  # for representation purposes
from torch_geometric.utils import (
    to_dense_adj,
    to_dense_batch,
)  # for representation purposes

# For LDS
from collections import Counter
from scipy.ndimage import convolve1d
from scipy.stats import gaussian_kde

# from utils import get_lds_kernel_window
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from ocpmodels.common.registry import registry
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

# from utils import get_lds_kernel_window
# from loss import weighted_mse_loss
from torchmetrics import F1Score
from torchmetrics.classification import BinaryF1Score
from data_model_utils import (
    MyDataModule,
    MyOwnDataset,
    collate_fn,
    data_list_collater,
    ocp_model,
    dimenet_model,
    cgcnn_model,
    gemnet_oc_segnn,
    schnet_segnn,
    schnet_model,
    schnet_onehot,
    schnet_onehot_detect,
    schnet_onehot_inmag,
    gemnet_oc_onehot,
    gemnet_oc_onehot_inmag,
    gemnet_oc_onehot_inmag_ff,
    schnet_onehot_inmag_FF
    
)
from pytorch_lightning.strategies import DDPStrategy
# wandb_logger = WandbLogger(
#    project="magmom",
#    name="gemnet_pure_energy_formation_per_atom",
# )
from pytorch_lightning.profilers import PyTorchProfiler
import pandas as pd
from tqdm import tqdm 
#import apex.optimizers as aoptim


class MagmomClassifier(LightningModule):
    def __init__(self, backbone, boundary, batch_size, energy_w = 1, magmom_w = 1, force_w=1, stress_w=0.1):
        super(MagmomClassifier, self).__init__()
        self.backbone = backbone
        self.boundary = boundary
        self.batch_size = batch_size
        self.energy_w = energy_w
        self.magmom_w = magmom_w
        self.force_w = force_w
        self.stress_w = stress_w

        self.save_hyperparameters()

        self.change_mat = (
            torch.tensor(
                [
                    [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                    [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                    [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                    [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                    [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                    [
                        -(6 ** (-0.5)),
                        0,
                        0,
                        0,
                        2 * 6 ** (-0.5),
                        0,
                        0,
                        0,
                        -(6 ** (-0.5)),
                    ],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                    [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
                ]
            )
            .detach()
            .to(self.device)
        )

    def forward(self, x, training=True):
        if not training:
            self.backbone.eval()
        print("Module is in training?", self.backbone.training)
        magmoms = self.backbone(x)
        return magmoms

#multiplies the base initial learning rate
#sqrt(global_batch_size/base_batch_size)
#square-root learning rate scaling rule

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)

        # optimizer = aoptim.FusedLAMB(self.parameters(), lr=1e-3,weight_decay=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", patience=50, factor=0.5
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        energy_preds, magmom_preds, force_preds, stress_iso, stress_aniso = self(batch)
        energy_loss = nn.L1Loss()(
            batch.corr_energy_per_atom,
            energy_preds.squeeze().type(batch.corr_energy_per_atom.dtype),
        ) 
        ind = torch.where(
           torch.abs(batch.y) >= self.boundary
        )  
        magmom_preds = magmom_preds[ind]
        magmom_loss = nn.L1Loss()(
           batch.y[ind], magmom_preds.squeeze().type(batch.y.dtype)
        )

        force_loss = nn.L1Loss()(batch.force, force_preds)

        batch_stress_decomposition = torch.einsum("ab, cb->ca", self.change_mat.to(self.device), batch.stress.reshape(-1, 9))
        stress_isotropic_target = batch_stress_decomposition[:, 0]
        stress_anisotropic_target = batch_stress_decomposition[:, 4:9]
        
        stress_loss = nn.L1Loss()(stress_iso.reshape(-1), stress_isotropic_target)
        stress_loss += nn.L1Loss()(stress_aniso.reshape(-1, 5), stress_anisotropic_target)
        
        energy_loss = self.energy_w*energy_loss
        magmom_loss = self.magmom_w*magmom_loss
        force_loss = self.force_w*force_loss
        stress_loss = self.stress_w*stress_loss

        loss = energy_loss + magmom_loss + force_loss  + stress_loss

        self.log("train_loss", loss.item(), on_epoch=True, batch_size=self.batch_size)
        self.log("magmom_loss_train", magmom_loss, on_epoch=True, batch_size=self.batch_size)
        return loss


    def on_train_epoch_end(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def validation_step(self, batch, batch_idx):
        energy_preds, magmom_preds, force_preds, stress_iso, stress_aniso  = self(batch)
        energy_loss = nn.L1Loss()(
            batch.corr_energy_per_atom,
            energy_preds.squeeze().type(batch.corr_energy_per_atom.dtype),
        ) 
        ind = torch.where(
           torch.abs(batch.y) >= self.boundary
        )  
        magmom_preds = magmom_preds[ind]
        magmom_loss = nn.L1Loss()(
           batch.y[ind], magmom_preds.squeeze().type(batch.y.dtype)
        )

        force_loss = nn.L1Loss()(batch.force, force_preds)

        batch_stress_decomposition = torch.einsum("ab, cb->ca", self.change_mat.to(self.device), batch.stress.reshape(-1, 9))
        stress_isotropic_target = batch_stress_decomposition[:, 0]
        stress_anisotropic_target = batch_stress_decomposition[:, 4:9]

        stress_loss = nn.L1Loss()(stress_iso.reshape(-1), stress_isotropic_target)
        stress_loss += nn.L1Loss()(stress_aniso.reshape(-1, 5), stress_anisotropic_target)

        energy_loss = self.energy_w*energy_loss
        magmom_loss = self.magmom_w*magmom_loss
        force_loss = self.force_w*force_loss
        stress_loss = self.stress_w*stress_loss

        val_loss = energy_loss + magmom_loss + force_loss + stress_loss

        self.log("val_loss", val_loss.item(), on_epoch=True, batch_size=self.batch_size)
        self.log("magmom_loss_val", magmom_loss, on_epoch=True, batch_size=self.batch_size)
        self.log("energy_loss_val", energy_loss, on_epoch=True, batch_size=self.batch_size)
        self.log("force_loss_val", force_loss, on_epoch=True, batch_size=self.batch_size)
        self.log("stress_loss_val", stress_loss, on_epoch=True, batch_size=self.batch_size)


        return {
            "val_loss": val_loss,
            #    "magmom_loss_val": magmom_loss_val
            # "num_correct": num_correct,
            # "num_samples": num_samples,
            # "yhat": preds,
            # "y": y,
        }
    

if __name__ == "__main__":

    file_name= "gemnet_FF"
    wandb_logger = WandbLogger(project="magmom_paper_new",
                                # id  = "qe0yyceb" ,
                                # resume = "must",
                                name=file_name)

    retrain = False
    preload = False

    num_devices = 1
    num_nodes = 1
    fast_dev_run= False
    model = gemnet_oc_onehot_inmag_ff
    data_path = "demo2"
    batch_size =  4 #!16 
    num_workers =  4
    energy_w = 1
    magmom_w = 1
    force_w = 1
    stress_w = 0.1
    boundary = 0

    profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler(f"tb_logs/{file_name}"),
        schedule = torch.profiler.schedule(skip_first=1, wait=1, warmup=1,
                                           active=5)
    )

    pl.seed_everything(1, workers=True)

    classifier_model = MagmomClassifier(backbone=model, 
                                        boundary=boundary, 
                                        batch_size=batch_size, energy_w=energy_w, 
                                        magmom_w=magmom_w,
                                        force_w = force_w,
                                        stress_w = stress_w
                                        )

    if retrain:
        checkpoint = "/pscratch/sd/w/wenxu/jobs/OCP/github_rep/ocp_mag_chgnet_paper_new/magmom_checkpoints/gemnet_onehot_inmag_91split_16mix_optimizer-epoch=505-step=252494-val_loss=0.0144-last.ckpt"

        classifier_model.load_state_dict(torch.load(checkpoint)["state_dict"])

    dataset_object = MyOwnDataset(structures=None, root=data_path, preload=preload)

    len_train_o = int(0.8 * len(dataset_object))
    len_val_o = int(0.1 * len(dataset_object))
    len_test_o = len(dataset_object) - len_train_o - len_val_o
    train_set_o, val_set_o, test_set_o = torch.utils.data.random_split(
       dataset_object, [len_train_o, len_val_o, len_test_o]
    )

    dmo = MyDataModule(
        batch_size=batch_size,
        train_set=train_set_o,
        val_set=val_set_o,
        test_set=test_set_o,
        train_collate_fn= data_list_collater, #collate_fn,  # To take abs and>=1
        val_collate_fn=data_list_collater,  #collate_fn,
        test_collate_fn=data_list_collater,  #collate_fn,
        num_workers=num_workers,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # need to implement recall score (tp / (tp + fn))
        mode="min",
        dirpath="magmom_checkpoints",
        save_last=True,
        filename= file_name +"-{epoch}-{step}-{val_loss:.4f}",
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = file_name + "-{epoch}-{step}-{val_loss:.4f}-last"

    learning_rate_callback = LearningRateMonitor(logging_interval="epoch")

    early_stopping_callback =  EarlyStopping(monitor='val_loss', 
            patience=1,
            verbose=True,
            mode='min'
            )

    trainer = pl.Trainer(
        #profiler=profiler,
        fast_dev_run = fast_dev_run,
        devices =num_devices,
        num_nodes = num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger,
        #gpus=1,
        # deterministic=True,
        max_epochs=1000,
        precision="16-mixed",
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value",
        # strategy="ddp",
        # logger=wandb_logger,
        callbacks=[learning_rate_callback, checkpoint_callback],#early_stopping_callback],
    )

    trainer.fit(
       classifier_model,
       dmo.train_dataloader(),
       dmo.val_dataloader(),
       #ckpt_path=checkpoint #!for resume
    )