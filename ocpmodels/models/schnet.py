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

from torch_geometric.nn.models.dimenet import (
    ResidualLayer,
)#!wx

from torch import nn#!wx
from ocpmodels.models.gemnet_oc.layers.base_layers import Dense, ResidualLayer
from ocpmodels.models.gemnet_oc.initializers import get_initializer

@registry.register_model("schnet")
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

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            # h = self.lin1(h)
            # h = self.act(h)
            # h = self.lin2(h)

            for _layer in self.out_mlp_energy: #!wx
                h_E = _layer(h) #!wx
            for _layer in self.out_mlp_mag: #!wx
                h_M = _layer(h) #!wx
                
            h_E = self.out_energy(h_E)
            h_M = self.out_mag(h_M)
            
            batch = torch.zeros_like(z) if batch is None else batch
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
            return energy, _M

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

