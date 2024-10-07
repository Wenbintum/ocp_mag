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


from typing import Callable, Dict, Optional, Tuple
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.inits import glorot_orthogonal
# from torch_geometric.nn.models.dimenet import (
#     ResidualLayer,
# )

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

class EmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        act: Callable,
        spin_hidden_channels=3,
    ):
        super().__init__()
        self.act = ScaledSiLU()

        # wx, this should be learnable
        self.emb = Embedding(95, hidden_channels)

        # #wx, onehot should not be learnable
        # self.spin_emb = torch.zeros(3, len(SPIN_ONEHOT_EMBEDDINGS[1]))
        # for i in range(100):
        #     self.embedding[i] = torch.tensor(embeddings[i + 1])
        self.spin_fc = Linear(3, spin_hidden_channels)

        #!wx, origin: 3*hidden_channels
        #!wx, need more layers?
        self.lin = Linear(
            hidden_channels + spin_hidden_channels, hidden_channels
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin.reset_parameters()
        self.spin_fc.reset_parameters()

    def forward(
        self, x: Tensor, s: Tensor) -> Tensor:
        # import pdb
        # pdb.set_trace()
        x = self.emb(x)  # samples*hidden_channels, samples in a batch

        # assume magmom value located at z
        # linear + spin
        # import pdb
        # pdb.set_trace()
        self.spin_emb = encode_to_one_hot(s)
        self.spin_emb = self.spin_emb.to(dtype=self.spin_fc.weight.dtype)
        spin_feat = self.spin_fc(self.spin_emb)


        return self.act(self.lin(
            torch.cat((x, spin_feat), dim=1)


        ))

@registry.register_model("schnet_SEGNN_onehot")
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

        segnn_gaussians: int = 50, #!wx
        act = "relu",
        num_layers_rbf = 2, 
        num_layers_gaussian = 2,

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

        
        #!wx    
        self._spinedge = SpinDistanceEdge(
            act="silu",
            cutoff=cutoff,
            num_layers_rbf=num_layers_rbf,
            num_layers_gaussian=num_layers_gaussian,
            num_gaussians=num_gaussians, #distance dim.
            segnn_gaussians=segnn_gaussians, #spin dim.
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
                    )
        
        self._reset_parameters() #!wx

    def _reset_parameters(self): #!wx
        self._spinedge.reset_parameters()


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


        magft_cat = data.magft #!wx
        magft_cat = magft_cat.to(edge_weight.device) #! check edge_weight
        j, i = edge_index #!data or batch?

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            #rbf distance and spin edge.

            edge_attr = self._spinedge(edge_weight, magft_cat, i, j) #wx

            #edge_attr = self.distance_expansion(edge_weight)

            h = self.combined_emb(z, data.magft[:, -1].long()) #segnn onhot


            #h = self.embedding(z) #!self.embedding = Embedding(100, hidden_channels, padding_idx=0) hidden 128 #spin-edge only
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)
            #!there is no edge info. just to update embedding. 
            #!in dimenet, output_block applied on x that is embedded node and edge.

            for _layer in self.out_mlp_energy: #!wx
                h_E = _layer(h) #!wx
            for _layer in self.out_mlp_mag: #!wx
                h_M = _layer(h) #!wx

            h_E = self.out_energy(h_E)
            h_M = self.out_mag(h_M)
            # h = self.lin1(h) #(h, h//2)
            # h = self.act(h)  #(h//2, h//2)
            # h = self.lin2(h) #(h//2, 1)

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(h_E, batch, dim=0, reduce=self.reduce)

        else:
            energy = super(SchNetWrap, self).forward(z, pos, batch)
            raise KeyError("shouldn't no pbc") #!wx 

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
    


#!mag introducing spin-distance edge.
class SpinDistanceEdge(torch.nn.Module):
    def __init__(
        self,
        act: str = "silu",
        cutoff: float = 5.0,
        num_layers_rbf=2,
        num_layers_gaussian=2,
        num_gaussians: int=50, #distance dim.
        segnn_gaussians:int = 50, #spin dim.
        ):
    
        #act = activation_resolver(act)
        super(SpinDistanceEdge, self).__init__()
        self.act = ScaledSiLU()


        #!mag gaussian
        self.distance_expansion = GaussianSmearing(
            start = 0.0,
            stop  = cutoff,
            num_gaussians=num_gaussians,
        )

        self.segnn_distance_expansion = GaussianSmearing(
            -1.5, 
            1.5, 
            segnn_gaussians) #!wx


        self.lin_spin = nn.Linear(
            segnn_gaussians + num_gaussians, num_gaussians, bias=True
        ) #[50 + 50] to [50]

        self.layers_rbf = torch.nn.ModuleList(
            [ResidualLayer(num_gaussians, activation=act) for _ in range(num_layers_rbf)]
        )

        self.layers_gaussian = torch.nn.ModuleList(
            [
                ResidualLayer(segnn_gaussians, activation=act)
                for _ in range(num_layers_gaussian)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):

        #!mag
        #!two gussian layers?
        glorot_orthogonal(self.lin_spin.weight, scale=2.0)

        self.lin_spin.bias.data.fill_(0)

        # for res_layer in self.layers_rbf:
        #     res_layer.reset_parameters()

        # for res_layer in self.layers_gaussian:
        #     res_layer.reset_parameters()


    def forward(self, dist, magft_cat, i, j):
        """_summary_

        Args:
            dist (_type_): _description_
            magft_cat (_type_): concatenated magmom features.
            i (_type_): indices
            j (_type_): indices

        Returns:
            _type_: _description_
        """

        # import pdb
        # pdb.set_trace()

        _distance_edge = self.distance_expansion(dist)
        #! compute (S_i ./dot S_j) and remove 1st dimension

        # import pdb
        # pdb.set_trace()
        
        _spin_edge     = self.segnn_distance_expansion(
            (magft_cat[i] * magft_cat[j]).sum(dim=1, keepdim=True)
        )

        #!mag, mlps(_rbf) and mlps(gaussian)
        for layer in self.layers_rbf:
            _distance_edge = layer(_distance_edge)

        for layer in self.layers_gaussian:
            _spin_edge = layer(_spin_edge)

        _spin_edge = _spin_edge.squeeze(1)

        #!stack |rbf, gaussian|
        rbf = torch.cat(
            (_distance_edge, _spin_edge), dim=1
        )  # [n_edges, n_rbf(rbf+gaussian)]

        #!maybe, lin_spin:  input dim: num_rbf+num_gaussians, output dim: num_rbf
        return self.act(self.lin_spin(rbf))

class ScaledSiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor