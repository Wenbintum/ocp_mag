"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import numpy as np
import torch

from .base_layers import Dense

from torch_geometric.nn.models.schnet import GaussianSmearing

class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Arguments
    ---------
    emb_size: int
        Atom embeddings size
    """

    def __init__(self, emb_size: int, num_elements: int) -> None:
        super().__init__()
        self.emb_size = emb_size

        self.embeddings = torch.nn.Embedding(num_elements, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(
            self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )

    def forward(self, Z) -> torch.Tensor:
        """
        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings
    and a subsequent dense layer.

    Arguments
    ---------
    atom_features: int
        Embedding size of the atom embedding.
    edge_features: int
        Embedding size of the input edge embedding.
    out_features: int
        Embedding size after the dense layer.
    activation: str
        Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features: int,
        edge_features: int,
        out_features: int,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(
            in_features, out_features, activation=activation, bias=False
        )

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        edge_index,
    ) -> torch.Tensor:
        """
        Arguments
        ---------
        h: torch.Tensor, shape (num_atoms, atom_features)
            Atom embeddings.
        m: torch.Tensor, shape (num_edges, edge_features)
            Radial basis in embedding block,
            edge embedding in interaction block.

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[edge_index[0]]  # shape=(nEdges, emb_size)
        h_t = h[edge_index[1]]  # shape=(nEdges, emb_size)

        m_st = torch.cat(
            [h_s, h_t, m], dim=-1
        )  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        return m_st



#SpinDistanceEdge
class SpinDistanceEdge(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings
    and a subsequent dense layer.

    Arguments
    ---------
    atom_features: int
        Embedding size of the atom embedding.
    edge_features: int
        Embedding size of the input edge embedding.
    out_features: int
        Embedding size after the dense layer.
    activation: str
        Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features: int,
        edge_features: int, #!wx change dim of edge features?
        out_features: int,
        activation: Optional[str] = None,
        #!new start and end of gaussian is not hp at the moment.
        segnn_gaussians: int = 50,  

    ) -> None:
        super().__init__()
        in_features = 2 * atom_features + edge_features + segnn_gaussians #!wx
        self.dense = Dense(
            in_features, out_features, activation=activation, bias=False
        )

        self.segnn_distance_expansion = GaussianSmearing(
            -1.5, 
            1.5, 
            segnn_gaussians) #!wx

    #!reset_parameters of gaussian?
    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        magft_cat,
        edge_index,
    ) -> torch.Tensor:
        """
        Arguments
        ---------
        h: torch.Tensor, shape (num_atoms, atom_features)
            Atom embeddings.
        m: torch.Tensor, shape (num_edges, edge_features)
            Radial basis in embedding block,
            edge embedding in interaction block.

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[edge_index[0]]  # shape=(nEdges, emb_size)
        h_t = h[edge_index[1]]  # shape=(nEdges, emb_size)

        #!wx
        _spin_edge     = self.segnn_distance_expansion(
            (magft_cat[edge_index[0]] * magft_cat[edge_index[1]]).sum(dim=1, keepdim=True)
        )

        m_st = torch.cat(
            [h_s, h_t, m, _spin_edge], dim=-1
        )  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        return m_st
