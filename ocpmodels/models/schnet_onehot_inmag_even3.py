"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch_geometric.nn import SchNet
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel

# from torch_geometric.nn.models.dimenet import (
#     ResidualLayer,
# )#!wx

from torch import nn, Tensor
from ocpmodels.models.gemnet_oc.layers.base_layers import Dense, ResidualLayer
from ocpmodels.models.gemnet_oc.initializers import get_initializer
from torch.nn import Embedding, Linear
from typing import Callable, Dict, Optional, Tuple, Union
from math import sqrt

def encode_to_one_hot(values):
    # Map the values to indices: -1 -> 0, 0 -> 1, 1 -> 2
    indices = values + 1

    # Number of classes for one-hot encoding (3 classes: -1, 0, 1)
    num_classes = 3

    # Perform one-hot encoding
    one_hot = torch.nn.functional.one_hot(indices, num_classes=num_classes)

    return one_hot

class EmbeddingBlock(nn.Module):
    def __init__( 
        self,
        #num_radial: int,
        hidden_channels: int,
        act: Callable,
        spin_hidden_channels=3,
        inmag_channels=1,
        ):
        super().__init__()
        #act = activation_resolver(act)
        self.act = ScaledSiLU() #act #!gemnet

        # Embedding for x
        self.emb = nn.Embedding(95, hidden_channels)

        # Fully connected layers for spin embeddings
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 4)

        # Multi-head attention for spin embeddings
        self.attention = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)

        # Linear layer for final combination
        self.lin = nn.Linear(hidden_channels + 4, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.lin.reset_parameters()
        self.attention._reset_parameters() 

    def process_spin(self, spin_emb, in_mag):
        spin_emb = torch.cat((spin_emb, in_mag), dim=1)  # Add sequence dimension for attention
        #spin_emb = self.attention(spin_emb, spin_emb, spin_emb)[0].squeeze(1)  # Remove sequence dimension after attention
        spin_emb = self.act(self.fc2(spin_emb))
        spin_emb = self.act(self.fc3(spin_emb))
        spin_emb = self.fc4(spin_emb)
        return spin_emb

    def forward(self, x: Tensor, s: Tensor, in_mag: Tensor) -> Tensor:
        x = self.emb(x)

        # Positive spin embedding
        spin_feat = self.process_spin(s.magft, in_mag.unsqueeze(1))

        # Negative spin embedding
        neg_spin_feat = self.process_spin(s.neg_magft, in_mag.unsqueeze(1))

        # Combine spin features
        s_feat = (spin_feat + neg_spin_feat) / 2

        # Final output
        combined = torch.cat((x, s_feat), dim=1)
        return self.act(self.lin(combined))

@registry.register_model("schnet_onehot_inmag_even3")
class SchNetWrap(SchNet, BaseModel):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = False,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        readout: str = "add",
    ) -> None:
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.reduce = readout
        super(SchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

        #out layer
        self.out_mlp_energy = torch.nn.ModuleList(
            [ResidualLayer(128, activation="silu") for _ in range(2)]
        ) #!wx schnet use 128 hidden channel by default

        self.out_mlp_mag = torch.nn.ModuleList(
            [ResidualLayer(128, activation="silu") for _ in range(2)]
        ) #!wx schnet use 128 hidden channel by default

        self.out_energy = Dense(
            128, num_targets, bias=False, activation=None
        )
#!wx
        self.out_mag = Dense(
            128, num_targets, bias=False, activation=None
        )

        output_init = "HeOrthogonal"
        out_initializer = get_initializer(output_init)
        self.out_energy.reset_parameters(out_initializer)
        self.out_mag.reset_parameters(out_initializer)

        self.combined_emb = EmbeddingBlock(
                    hidden_channels=128,
                    act="silu",
                    spin_hidden_channels=3,
                    inmag_channels=1,
                    )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)


        # breakpoint()
        #default spin vecotr 0.1
        # magft_cat = data.magft #!wx
        # magft_cat = magft_cat.to(z.device) #!wx


        #! test different spin critier for determining spin vector
        # magft_cat = torch.zeros(data.y.size(dim=0), 3)
        # magft_cat[data.y > 0.4, 2] = 1
        # magft_cat[data.y < -0.4, 2] = -1
        # magft_cat = magft_cat.to(z.device)
        
        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            edge_attr = self.distance_expansion(edge_weight)

            #breakpoint()
            h = self.combined_emb(z, data, data.inmag)
            #h = self.combined_emb(z, data.magft, data.inmag)


            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            # h = self.lin1(h)
            # h = self.act(h)
            # h = self.lin2(h)
            h_E, h_M = h, h
            for _layer in self.out_mlp_energy: #!wx
                h_E = _layer(h_E) #!wx     #This looks only used the last h_E?
            for _layer in self.out_mlp_mag: #!wx
                h_M = _layer(h_M) #!wx
                
            h_E = self.out_energy(h_E)
            h_M = self.out_mag(h_M)
            #!project 128 to 1. number equals to nodes.
            #!h_E should be 128 x N -> 1 x N
            batch = torch.zeros_like(z) if batch is None else batch

            #!energy is sum of these nodes.
            energy = scatter(h_E, batch, dim=0, reduce=self.reduce) #!wx
        else:
            energy = super(SchNetWrap, self).forward(z, pos, batch)
        return energy, h_M

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy, _M = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, _M, forces
        else:
            #relu_nonconstrain
            M_pred_vec = torch.where(data.tmp != 0, _M * data.tmp, _M)
            return energy, M_pred_vec

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

class ScaledSiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor
