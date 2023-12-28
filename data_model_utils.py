from ocpmodels.common.utils import setup_imports
import torch
import itertools
import json
import hashlib
import os
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
from pytorch_lightning import LightningModule, LightningDataModule
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

setup_imports()


class MyDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_set,
        val_set,
        test_set,
        train_collate_fn=None,
        val_collate_fn=None,
        test_collate_fn=None,
    ):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn
        self.test_collate_fn = test_collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            collate_fn=self.train_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8,
            collate_fn=self.val_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8,
            collate_fn=self.test_collate_fn,
        )


class MyOwnDataset(Dataset):
    def __init__(self, structures=None, root="disk_data"):
        self.structures = structures
        super(MyOwnDataset, self).__init__(root=root)
        # if not os.path.exists("disk_data"):
        #    os.makedirs("disk_data")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return sorted(os.listdir(self.processed_dir))

    def __len__(self):
        return len(self.processed_file_names)

    #        return len(self.structures)

    def __getitem__(self, idx):
        # try:
        #    data = torch.load(f"{self.processed_dir}/data_{idx}.pt")
        # except FileNotFoundError:
        #    return None
        data = torch.load(f"{self.processed_dir}/data_{idx}.pt")
        return data

    def process(self):
        count = 0
        for i, structure in enumerate(self.structures):
            try:
                print(
                    f"Encoding sample {i+1:5d}/{len(self.structures):5d}",
                    end="\r",
                    flush=True,
                )
                analyzed_structure = pg.CollinearMagneticStructureAnalyzer(
                    structure["structure"]
                )
                order_list_mp.append(analyzed_structure.ordering)
                structures_list_mp.append(structure["structure"])
                formula_list_mp.append(structure["pretty_formula"])
                id_list_mp.append(structure["material_id"])
                sites_list.append(structure["nsites"])
                node_x = embedding[
                    torch.LongTensor(structure["structure"].atomic_numbers)
                ]
                # FIXME: Need to check that not all values zero for at least one row in the node feature matrix - means we are missing encoding data
                y = torch.tensor(structure["structure"].site_properties["magmom"])
                try:
                    print(
                        node_x[
                            torch.where(node_x > 0)[0], torch.where(node_x > 0)[1]
                        ].reshape(node_x.size(0), 3)
                    )
                except RuntimeError:
                    print("Skipping since can't encode...")
                    continue
                if torch.where((node_x == 0.0).all(dim=1))[0].nelement() == 0:
                    data = Data(
                        x=node_x,
                        # Rs_in=None,
                        pos=torch.tensor(structure["structure"].cart_coords.copy()),
                        cell=torch.tensor(structure["structure"].lattice.matrix.copy()),
                        r_max=params["max_radius"],
                        y=y,
                        ordering=analyzed_structure.ordering,
                        n_norm=n_norm,
                    )
                    torch.save(data, f"{self.processed_dir}/data_{i}.pt")
                #                input = torch.zeros(len(struct), 3 * len_element)
                #                for j, site in enumerate(struct):
                #                    input[j, int(element(str(site.specie)).atomic_number)] = element(
                #                        str(site.specie)
                #                    ).atomic_radius
                #                    # input[j, len_element + int(element(str(site.specie)).atomic_number) +1] = element(str(site.specie)).atomic_weight
                #                    input[
                #                        j,
                #                        len_element + int(element(str(site.specie)).atomic_number) + 1,
                #                    ] = element(str(site.specie)).en_pauling
                #                    input[
                #                        j,
                #                        2 * len_element
                #                        + int(element(str(site.specie)).atomic_number)
                #                        + 1,
                #                    ] = element(str(site.specie)).dipole_polarizability
                #                    #                data.append(
                #                    #                    DataPeriodicNeighbors(
                #                    #                        x=input,
                #                    #                        Rs_in=None,
                #                    #                        pos=torch.tensor(struct.cart_coords.copy()),
                #                    #                        lattice=torch.tensor(struct.lattice.matrix.copy()),
                #                    #                        r_max=params["max_radius"],
                #                    #                        y=(torch.tensor([y_values[i]])).to(torch.long),
                #                    #                        n_norm=n_norm,
                #                    #                    )
                #                    #                )
                #
                #                    torch.save(
                #                        DataPeriodicNeighbors(
                #                            x=input,
                #                            Rs_in=None,
                #                            pos=torch.tensor(struct.cart_coords.copy()),
                #                            lattice=torch.tensor(struct.lattice.matrix.copy()),
                #                            r_max=params["max_radius"],
                #                            y=(torch.tensor([y_values[i]])).to(torch.long),
                #                            n_norm=n_norm,
                #                        ),
                #                        f"{self.processed_dir}/data_{i}.pt",
                #                    )
                count += 1
            except Exception as e:
                print(f"Error: {count} {e}", end="\n")
                # count += 1
                continue


####### COLLATE_FXNS ##########
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch_geometric.data.Batch.from_data_list(batch)


dimenet_model = registry.get_model_class("dimenetplusplus_SEGNN_onehotL")(
    None, -1, 1, regress_forces=False
)
cgcnn_model = registry.get_model_class("cgcnn")(
    None,
    -1,
    1,
    regress_forces=False,
)
ocp_model = registry.get_model_class("gemnet_oc")(
    None,
    -1,
    1,  # out_channels
    num_spherical=7,
    num_radial=128,
    num_blocks=4,
    emb_size_atom=256,
    emb_size_edge=512,
    emb_size_trip_in=64,
    emb_size_trip_out=64,
    emb_size_quad_in=32,
    emb_size_quad_out=32,
    emb_size_aint_in=64,
    emb_size_aint_out=64,
    emb_size_rbf=16,
    emb_size_cbf=16,
    emb_size_sbf=32,
    num_before_skip=2,
    num_after_skip=2,
    num_concat=1,
    num_atom=3,
    num_output_afteratom=3,
    num_atom_emb_layers=2,
    num_global_out_layers=2,
    regress_forces=False,
    direct_forces=True,
    use_pbc=True,
    cutoff=12.0,
    cutoff_qint=12.0,
    cutoff_aeaint=12.0,
    cutoff_aint=12.0,
    max_neighbors=30,
    max_neighbors_qint=8,
    max_neighbors_aeaint=20,
    max_neighbors_aint=1000,
    rbf={"name": "gaussian"},
    envelope={"name": "polynomial", "exponent": 5},
    cbf={"name": "spherical_harmonics"},
    sbf={"name": "legendre_outer"},
    extensive=False,
    forces_coupled=False,
    output_init="HeOrthogonal",
    activation="silu",
    quad_interaction=True,
    atom_edge_interaction=True,
    edge_atom_interaction=True,
    atom_interaction=True,
    num_elements=100,
    # otf_graph=True,
    qint_tags=[0, 1, 2],  # calculate quadruplet interactions for all types of nodes
)


class EmbeddingData(DS):
    def __init__(self, filename):
        super(EmbeddingData, self).__init__()
        # synthetic_data = torch.load("filename")
        # syn_emb_data = synthetic_data["embeddings"]
        # syn_magmoms = synthetic_data["magmoms"]
        self.data = torch.load(filename)
        self.all_emb_data = self.data["node_embeddings"]
        # self.all_magmom_mask = self.data["magmom_mask"]
        # self.all_magmom_mask_targs = self.data["magmom_mask_targs"]
        self.all_magmoms = self.data["magmom_labels"].cpu().squeeze()
        # self.gemnet_indices = torch.arange(self.all_emb_data.size(0))

        indices = torch.where(
            (self.all_magmoms >= 0.1) | (self.all_magmoms <= -0.1)
        )  # Note if not using targs then some zero values will seep in by default, which is not a bad thing

        self.int_emb_data = self.all_emb_data[indices]
        # self.int_emb_data = torch.vstack((self.int_emb_data, syn_emb_data))

        self.int_magmoms = self.all_magmoms[indices]
        # self.int_magmoms = torch.hstack((self.int_magmoms, syn_magmoms.squeeze()))
        # self.int_indices = self.gemnet_indices[indices]
        # self.syn_indices = torch.tensor(-1).expand_as(syn_magmoms).squeeze()
        # self.int_indices = torch.hstack((self.int_indices, self.syn_indices))

    #        # Remove outliers??
    #        self.int_magmoms = torch.abs(self.int_magmoms)
    #        inlier_indices = torch.where((self.int_magmoms >= 1))
    #        self.int_magmoms = self.int_magmoms[inlier_indices]
    #        self.int_emb_data = self.int_emb_data[inlier_indices]

    def __getitem__(self, idx):
        # return self.all_emb_data[idx], self.all_magmoms[idx], self.all_indices[idx]
        return self.int_emb_data[idx], self.int_magmoms[idx]  # , self.int_indices[idx]

    def __len__(self):
        return self.int_magmoms.size(0)


class FFBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(FFBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.lin = nn.Linear(self.in_feat, self.out_feat)
        self.bnorm = nn.BatchNorm1d(self.out_feat)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        out = nn.Sequential(self.lin, self.bnorm, self.act)(x)
        return out


class EmbeddingRegressor(pl.LightningModule):
    def __init__(self):
        super(EmbeddingRegressor, self).__init__()
        # self.fcs = nn.Sequential(
        #    nn.Linear(256, 512),
        #    nn.BatchNorm1d(512),
        #    nn.LeakyReLU(),
        #    # nn.Dropout(0),
        #    nn.Linear(512, 1024),
        #    nn.BatchNorm1d(1024),
        #    nn.LeakyReLU(),
        #    # nn.Dropout(0),
        #    nn.Linear(1024, 1024),
        #    nn.BatchNorm1d(1024),
        #    nn.LeakyReLU(),
        #    # nn.Dropout(0.2),
        #    nn.Linear(1024, 512),
        #    nn.BatchNorm1d(512),
        #    nn.LeakyReLU(),
        #    # nn.Dropout(0),
        #    nn.Linear(512, 256),
        #    nn.BatchNorm1d(256),
        #    nn.LeakyReLU(),
        #    # nn.Dropout(0.2),
        #    nn.Linear(256, 128),
        #    nn.BatchNorm1d(128),
        #    nn.LeakyReLU(),
        #    # nn.Dropout(0),
        #    nn.Linear(128, 64),
        #    nn.BatchNorm1d(64),
        #    nn.LeakyReLU(),
        #    # nn.Dropout(0),
        #    nn.Linear(64, 16),
        #    nn.BatchNorm1d(16),
        #    nn.Dropout(0.2),  # FIXME: This was set to default p = 0.5
        #    nn.LeakyReLU(),
        #    nn.Linear(16, 1),
        # )
        self.block1 = FFBlock(256, 512)
        self.block2 = FFBlock(512, 1024)
        self.block3 = FFBlock(1024, 1024)
        self.block4 = FFBlock(1024, 512)
        self.block5 = FFBlock(512, 256)
        self.block6 = FFBlock(256, 128)
        self.block7 = FFBlock(128, 64)
        self.block8 = FFBlock(64, 16)
        self.block9 = FFBlock(16, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # magmoms = self.fcs(x)
        x1 = self.dropout(self.block1(x))
        x2 = self.dropout(self.block2(x1))
        x3 = self.dropout(self.block3(x2) + x2)
        x4 = self.dropout(self.block4(x3) + x1)
        x5 = self.dropout(self.block5(x4) + x)
        x6 = self.dropout(self.block6(x5))
        x7 = self.dropout(self.block7(x6))
        x8 = self.dropout(self.block8(x7))
        magmoms = self.block9(x8)
        return magmoms

    def training_step(self, batch, batch_idx):
        preds = self(batch[0]).squeeze()
        # focus_magmoms = batch[1].cpu().numpy()
        # bins = np.linspace(min(focus_magmoms), max(focus_magmoms), 200)
        # bin_index_per_label = np.digitize(focus_magmoms, bins)
        # Nb = max(bin_index_per_label) + 1
        # num_samples_of_bins = dict(Counter(bin_index_per_label))
        # emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
        # lds_kernel_window = get_lds_kernel_window(kernel="gaussian", ks=5, sigma=2)
        # eff_label_dist = convolve1d(
        #    np.array(emp_label_dist), weights=lds_kernel_window, mode="constant"
        # )
        ### a = np.arange(len(emp_label_dist))
        ### ax5.hist(a, len(a), weights=emp_label_dist)
        # eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        # weights = [np.float32(1 / x) for x in eff_num_per_label]
        ## scaling = len(weights) / np.sum(weights)
        # scaling = 1
        # final_weights = torch.tensor([scaling * w for w in weights]).to(self.device)
        # final_weights = torch.ones_like(final_weights)
        ## loss = nn.MSELoss()(preds, batch[1].type(preds.dtype))
        ## pred_magmoms = self.regressor_model(batch, epoch=None)[0].squeeze() # cgcnn
        # loss = (torch.abs((preds - batch[1])) * final_weights).sum() / preds.size(0)
        # loss = (((preds - batch[1]) ** 2)).sum() / preds.size()[0]
        loss = nn.L1Loss()(preds, torch.abs(batch[1]).type(preds.dtype))  # NO SIGN
        # loss = nn.L1Loss()(preds, batch[1].type(preds.dtype))  # SIGN
        # loss = nn.HuberLoss(delta=0.5)(preds, batch[1].type(preds.dtype))
        self.log("training_loss", loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        preds = self(batch[0]).squeeze()
        labels = torch.abs(batch[1])  # NO SIGN
        # labels = batch[1]  # SIGN
        #        loss = nn.MSELoss()(preds, labels.type(preds.dtype))
        loss = nn.L1Loss()(preds, labels.type(preds.dtype))
        self.log("embedding_val_loss", loss, on_epoch=True)
        return {"embedding_val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=1e-6
        )  # Switch to 1e-4 when finetune
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", patience=100, factor=0.75
                ),
                # "scheduler": CosineAnnealingLR(optimizer, 10, eta_min=1e-4),
                #        "scheduler": CyclicLR(
                #            optimizer, base_lr=1e-5, max_lr=1e-3, cycle_momentum=False
                #        ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "embedding_val_loss",
            },
        }
