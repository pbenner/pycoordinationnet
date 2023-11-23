## Copyright (C) 2023 Philipp Benner
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## ----------------------------------------------------------------------------

import torch

from typing import Union

from torch_geometric.data    import Data
from torch_geometric.nn      import Sequential, GraphConv, CGConv, HeteroConv, ResGatedGraphConv, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing  import Adj, OptTensor, PairTensor

from .features_coding import NumOxidations, NumGeometries

from .model_layers     import TorchStandardScaler, ModelDense, ElementEmbedder, RBFEmbedding, PaddedEmbedder
from .model_gnn_config import DefaultGraphCoordinationNetConfig

## ----------------------------------------------------------------------------

class IdConv(MessagePassing):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None) -> torch.Tensor:

        return x

## ----------------------------------------------------------------------------

class ModelGraphCoordinationNet(torch.nn.Module):
    def __init__(self,
        # Specify model components
        model_config = DefaultGraphCoordinationNetConfig,
        # Options for dense layers
        layers = [512, 128, 64, 1], **kwargs):

        super().__init__()

        # Feature dimensions
        dim_element   = model_config['dim_element']
        dim_oxidation = model_config['dim_oxidation']
        dim_geometry  = model_config['dim_geometry']
        dim_csm       = model_config['dim_csm']
        dim_distance  = model_config['dim_distance']
        dim_angle     = model_config['dim_angle']

        dim_site   = dim_element + dim_oxidation
        dim_ce     = dim_element + dim_oxidation + dim_geometry + dim_csm
        dim_ligand = dim_element + dim_oxidation

        if model_config['distances']:
            dim_ligand += dim_distance

        if model_config['angles']:
            dim_ligand += dim_angle

        if model_config['num_convs'] is None:
            model_config['num_convs'] = 2

        if model_config['conv_type'] is None:
            model_config['conv_type'] = 'CGConv'

        # The model config determines which components of the model
        # are active
        self.model_config        = model_config
        # Optional scaler of model outputs (predictions)
        self.scaler_outputs      = TorchStandardScaler(layers[-1])

        # RBF encoder
        self.rbf_csm             = RBFEmbedding(0.0, 1.0, bins=model_config['bins_csm']     , edim=dim_csm)
        self.rbf_distances_1     = RBFEmbedding(0.0, 1.0, bins=model_config['bins_distance'], edim=dim_distance)
        self.rbf_distances_2     = RBFEmbedding(0.0, 1.0, bins=model_config['bins_distance'], edim=dim_distance)
        self.rbf_angles          = RBFEmbedding(0.0, 1.0, bins=model_config['bins_angle']   , edim=dim_angle)

        # Embeddings
        self.embedding_element   = ElementEmbedder(dim_element, from_pretrained=True, freeze=True)
        self.embedding_oxidation = torch.nn.Embedding(NumOxidations, dim_oxidation)
        self.embedding_geometry  = PaddedEmbedder(NumGeometries, dim_geometry)

        self.activation          = torch.nn.ELU(inplace=True)

        sequential_layers = []

        for _ in range(model_config['num_convs']):

            # Add convolution operation

            if model_config['conv_type'] == 'CGConv':

                sequential_layers.append(
                    (HeteroConv({
                        ('site'  , '*', 'site'  ): IdConv(),
                        ('ligand', '*', 'ce'    ): CGConv((dim_ligand, dim_ce), dim_distance, add_self_loops=True),
                        ('ce'    , '*', 'ligand'): CGConv((dim_ce, dim_ligand), dim_distance, add_self_loops=True),
                    }), 'x, edge_index, edge_attr -> x')
                )

            elif model_config['conv_type'] == 'ResGatedGraphConv':

                sequential_layers.append(
                    (HeteroConv({
                        ('site'  , '*', 'site'  ): IdConv(),
                        ('ligand', '*', 'ce'    ): ResGatedGraphConv((dim_ligand, dim_ce), dim_ce    , edge_dim=dim_distance, add_self_loops=True),
                        ('ce'    , '*', 'ligand'): ResGatedGraphConv((dim_ce, dim_ligand), dim_ligand, edge_dim=dim_distance, add_self_loops=True),
                    }), 'x, edge_index, edge_attr -> x')
                )

            else:
                raise ValueError('Invalid conv_type specified')

            # Add activation function
            sequential_layers.append(
                (lambda x: { k : v if k == 'site' else self.activation(v) for k, v in x.items()}, 'x -> x'),
            )

        sequential_layers += [
            # Mixing layer
            # -------------------------------------------------------------------------------------------
            (HeteroConv({
                ('ce', '*', 'site'): GraphConv((dim_ce, dim_site), dim_site, add_self_loops=True, bias=False),
            }, aggr='mean'), 'x, edge_index -> x'),
            # Apply activation
            (lambda x: { k : self.activation(v) for k, v in x.items()}, 'x -> x'),
            # Output layer
            # -------------------------------------------------------------------------------------------
            # Apply mean pooling
            (lambda x, batch: { k : global_mean_pool(v, batch[k]) for k, v in x.items() }, 'x, batch -> x'),
            # Extract only site features
            (lambda x: x['site'], 'x -> x')
        ]

        # Core graph network
        self.layers = Sequential('x, edge_index, edge_attr, batch', sequential_layers)

        # Final dense layer
        self.dense = ModelDense([dim_site] + layers, **kwargs)

    def forward(self, x_input):

        x_site = torch.cat((
            self.embedding_element  (x_input['site'].x['elements'  ]),
            self.embedding_oxidation(x_input['site'].x['oxidations']),
            ), dim=1)

        x_ce = torch.cat((
            self.embedding_element  (x_input['ce'].x['elements'  ]),
            self.embedding_oxidation(x_input['ce'].x['oxidations']),
            self.embedding_geometry (x_input['ce'].x['geometries']),
            self.rbf_csm            (x_input['ce'].x['csms'      ]),
            ), dim=1)

        x_ligand = torch.cat((
            self.embedding_element  (x_input['ligand'].x['elements'  ]),
            self.embedding_oxidation(x_input['ligand'].x['oxidations']),
            ), dim=1)

        # Add optional features
        if self.model_config['distances']:
            x_ligand = torch.cat((
                x_ligand,
                self.rbf_distances_1(x_input['ligand'].x['distances']),
                ), dim=1)

        if self.model_config['angles']:
            x_ligand = torch.cat((
                x_ligand,
                self.rbf_angles(x_input['ligand'].x['angles']),
                ), dim=1)

        # Concatenate embeddings to yield a single feature vector per node
        x = {
            'site': x_site, 'ce': x_ce, 'ligand': x_ligand,
        }
        edge_attr_dict = {
            ('ligand', '*', 'ce'): self.rbf_distances_2(x_input['ligand', '*', 'ce'].edge_attr),
            ('ce', '*', 'ligand'): self.rbf_distances_2(x_input['ce', '*', 'ligand'].edge_attr),
        }
        # Propagate features through graph network
        x = self.layers(x, x_input.edge_index_dict, edge_attr_dict, x_input.batch_dict)
        # Apply final dense layer
        x = self.dense(x)
        # Apply inverse transformation
        x = self.scaler_outputs.inverse_transform(x)

        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters_grouped(self):
        return { 'all': self.parameters() }
